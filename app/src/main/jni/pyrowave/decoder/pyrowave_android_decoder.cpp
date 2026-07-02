/**
 * @file pyrowave_android_decoder.cpp
 * @brief JNI native PyroWave decoder for moonlight-android.
 *
 * Java <-> native bridge (class com.limelight.binding.video.PyroWaveDecoder):
 *   long nativeSetup(Surface surface, int width, int height)  -> handle (0 on fail)
 *   int  nativeSubmit(long handle, byte[] data, int length)   -> 0 ok, -1 need IDR
 *   void nativeCleanup(long handle)
 *
 * Decode runs on a private Vulkan device (PyroWave::Decoder). The decoded YCbCr
 * planes are converted to RGBA by a compute shader (yuv2rgba, BT.709 limited,
 * matching the encoder), blitted onto an acquired swapchain image, and presented
 * on the GPU - no CPU readback and no ANativeWindow software post.
 *
 * NOT compiled/tested here (no Android NDK/Vulkan). Expect on-device iteration;
 * the YCbCr->RGB matrix, the decode-unit framing, and the layout barriers around
 * Decoder::decode are the most likely things to need adjustment.
 */
#include <jni.h>
#include <android/log.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <time.h>
#include <utility>
#include <vector>

#include <vulkan/vulkan_raii.hpp>
#include "pyrowave/pyrowave_decoder.h"
#include "pyrowave/pyrowave_common.h"  // PyroWave::load_shader
#include "vk/allocation.h"
#include "pyrowave_vk_android.h"

#define PWLOG(prio, ...) __android_log_print(prio, "PyroWaveDec", __VA_ARGS__)

namespace {

  constexpr int DR_OK = 0;
  constexpr int DR_NEED_IDR = -1;

  image_allocation make_plane_image(vk::raii::Device &device, uint32_t w, uint32_t h) {
    vk::ImageCreateInfo info {
      .imageType = vk::ImageType::e2D,
      .format = vk::Format::eR8Unorm,
      .extent = {.width = w, .height = h, .depth = 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
               vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
    };
    return image_allocation(device, info, {.usage = VMA_MEMORY_USAGE_AUTO}, "pyrowave plane");
  }

  vk::raii::ImageView make_plane_view(vk::raii::Device &device, vk::Image img) {
    return device.createImageView(vk::ImageViewCreateInfo {
      .image = img,
      .viewType = vk::ImageViewType::e2D,
      .format = vk::Format::eR8Unorm,
      .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1},
    });
  }

  struct decoder_state {
    std::shared_ptr<pyrowave_vk::context> ctx;
    std::unique_ptr<PyroWave::Decoder> decoder;
    std::unique_ptr<PyroWave::DecoderInput> input;

    int width = 0, height = 0;
    image_allocation img_y, img_cb, img_cr;
    vk::raii::ImageView view_y = nullptr, view_cb = nullptr, view_cr = nullptr;
    bool images_initialized = false;
    vk::raii::CommandPool cmd_pool = nullptr;
    vk::raii::CommandBuffer cmd = nullptr;
    vk::raii::Fence fence = nullptr;

    ANativeWindow *window = nullptr;

    // Swapchain present: a compute shader (yuv2rgba) converts the decoded planes
    // into an offscreen RGBA image which is blitted onto the acquired swapchain
    // image and presented. No CPU readback, no ANativeWindow software post.
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::SwapchainKHR swapchain = nullptr;
    std::vector<vk::Image> swap_images;
    vk::Format swap_format = vk::Format::eUndefined;
    vk::Extent2D swap_extent {};
    // When the surface allows it (R8G8B8A8 + STORAGE usage), yuv2rgba writes the
    // converted frame straight into the acquired swapchain image (scaling via the
    // shader's UV sampling), skipping the offscreen image + blit copy. Otherwise we
    // fall back to converting into img_rgba and blitting.
    bool direct_present = false;
    std::vector<vk::raii::ImageView> swap_storage_views;  ///< per-swapchain-image, direct path
    image_allocation img_rgba;              ///< offscreen R8G8B8A8 (compute target; blit path)
    vk::raii::ImageView rgba_view = nullptr;
    vk::raii::Sampler sampler = nullptr;
    vk::raii::DescriptorSetLayout dsl = nullptr;
    vk::raii::PipelineLayout pl = nullptr;
    vk::raii::Pipeline pipe = nullptr;
    vk::raii::DescriptorPool dpool = nullptr;
    vk::raii::DescriptorSet dset = nullptr;
    // Ping-pong present sync so one frame can be in GPU flight while the receive
    // thread depacketizes the next (pipelining). Index by frame parity.
    vk::raii::Semaphore acquire_sem[2] = {nullptr, nullptr};
    vk::raii::Semaphore present_sem[2] = {nullptr, nullptr};
    bool have_pending = false;   ///< a submitted-but-not-yet-waited frame exists
    uint32_t parity = 0;

    // Phase-offset pacing (pyrofling-style): the value the JNI layer hands to
    // Java, which forwards it to the host via LiSendPhaseOffset.
    //
    // Preferred source: VK_GOOGLE_display_timing. Each present carries an ID;
    // vkGetPastPresentationTimingGOOGLE later reports the frame's ACTUAL
    // present time (CLOCK_MONOTONIC). The submit->present margin, relative to
    // a small target guard band, is the phase error: positive (frame ready
    // too early) => host should capture later; negative => earlier. This
    // phase-locks the host's capture clock to this display - smoothness with
    // no added buffering - and works under any present mode.
    //
    // Fallback (no display timing): time blocked in acquireNextImage, which
    // is only meaningful under FIFO backpressure.
    int last_phase_offset_us = 0;
    bool first_frame_done = false;  ///< gate the cold-start acquire out of phase pacing
    bool use_display_timing = false;
    uint32_t present_counter = 0;
    std::deque<std::pair<uint32_t, int64_t>> pending_present_times;  ///< (presentID, submit CLOCK_MONOTONIC ns)

    static int64_t now_monotonic_ns() {
      struct timespec ts {};
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return int64_t(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
    }

    // Drain vkGetPastPresentationTimingGOOGLE and turn the newest entry into a
    // phase error for the host. Called on the decoder thread around present.
    void poll_display_timing() {
      constexpr int64_t kTargetMarginUs = 5000;
      constexpr int64_t kMaxErrUs = 15000;
      std::vector<vk::PastPresentationTimingGOOGLE> timings;
      try {
        timings = swapchain.getPastPresentationTimingGOOGLE();
      } catch (const vk::SystemError &) {
        return;  // transient (e.g. out-of-date); keep the previous value
      }
      for (const auto &t : timings) {
        // Find (and consume up to) the matching submit record.
        int64_t submit_ns = -1;
        while (!pending_present_times.empty()) {
          auto front = pending_present_times.front();
          if (int32_t(front.first - t.presentID) > 0) {
            break;  // timing for an ID we no longer track
          }
          pending_present_times.pop_front();
          if (front.first == t.presentID) {
            submit_ns = front.second;
            break;
          }
        }
        if (submit_ns < 0) {
          continue;
        }
        int64_t err_us = (int64_t(t.actualPresentTime) - submit_ns) / 1000 - kTargetMarginUs;
        if (err_us > kMaxErrUs) err_us = kMaxErrUs;
        if (err_us < -kMaxErrUs) err_us = -kMaxErrUs;
        last_phase_offset_us = (int) err_us;
      }
    }

    // Wait for the previously submitted frame's GPU work to finish. Called before
    // reusing the shared decode/convert resources and the input buffers.
    void wait_prev() {
      if (have_pending) {
        (void) ctx->device().waitForFences(*fence, true, UINT64_MAX);
        have_pending = false;
      }
    }

    ~decoder_state() {
      if (ctx) {
        try { (void) ctx->device().waitIdle(); } catch (...) {}
      }
      if (window) {
        ANativeWindow_release(window);
      }
    }

    struct ConvPush { int32_t w, h; float inv_w, inv_h; int32_t src_w, src_h; int32_t sharp; };
    bool swapchain_stale = false;  ///< out-of-date/suboptimal seen; recreate before next frame

    bool init() {
      try {
        auto &device = ctx->device();
        uint32_t cw = (uint32_t(width) + 1) / 2;
        uint32_t ch = (uint32_t(height) + 1) / 2;

        img_y = make_plane_image(device, width, height);
        img_cb = make_plane_image(device, cw, ch);
        img_cr = make_plane_image(device, cw, ch);
        view_y = make_plane_view(device, img_y);
        view_cb = make_plane_view(device, img_cb);
        view_cr = make_plane_view(device, img_cr);

        cmd_pool = vk::raii::CommandPool(device, vk::CommandPoolCreateInfo {
          .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
          .queueFamilyIndex = ctx->caps().compute_queue_family});
        cmd = std::move(device.allocateCommandBuffers(vk::CommandBufferAllocateInfo {
          .commandPool = *cmd_pool, .commandBufferCount = 1})[0]);
        fence = vk::raii::Fence(device, vk::FenceCreateInfo {});
        for (int i = 0; i < 2; ++i) {
          acquire_sem[i] = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo {});
          present_sem[i] = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo {});
        }

        // --- Surface + swapchain on the Android window (set before init()) ---
        surface = vk::raii::SurfaceKHR(ctx->instance(), vk::AndroidSurfaceCreateInfoKHR {.window = window});
        auto &phys = ctx->physical_device();
        if (!phys.getSurfaceSupportKHR(ctx->caps().compute_queue_family, *surface)) {
          PWLOG(ANDROID_LOG_ERROR, "queue family does not support present");
          return false;
        }
        if (!create_swapchain()) {
          return false;
        }

        // --- Offscreen R8G8B8A8 compute target (matches the shader's rgba8) ---
        vk::ImageCreateInfo ri {
          .imageType = vk::ImageType::e2D, .format = vk::Format::eR8G8B8A8Unorm,
          .extent = {.width = (uint32_t) width, .height = (uint32_t) height, .depth = 1},
          .mipLevels = 1, .arrayLayers = 1,
          .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc};
        img_rgba = image_allocation(device, ri, {.usage = VMA_MEMORY_USAGE_AUTO}, "pyrowave rgba");
        rgba_view = device.createImageView(vk::ImageViewCreateInfo {
          .image = img_rgba, .viewType = vk::ImageViewType::e2D, .format = vk::Format::eR8G8B8A8Unorm,
          .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}});

        // --- yuv2rgba compute pipeline ---
        sampler = vk::raii::Sampler(device, vk::SamplerCreateInfo {
          .magFilter = vk::Filter::eLinear, .minFilter = vk::Filter::eLinear,
          .mipmapMode = vk::SamplerMipmapMode::eNearest,
          .addressModeU = vk::SamplerAddressMode::eClampToEdge,
          .addressModeV = vk::SamplerAddressMode::eClampToEdge,
          .addressModeW = vk::SamplerAddressMode::eClampToEdge});
        std::array<vk::DescriptorSetLayoutBinding, 5> binds {{
          {.binding = 0, .descriptorType = vk::DescriptorType::eSampledImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
          {.binding = 1, .descriptorType = vk::DescriptorType::eSampledImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
          {.binding = 2, .descriptorType = vk::DescriptorType::eSampledImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
          {.binding = 3, .descriptorType = vk::DescriptorType::eSampler,      .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
          {.binding = 4, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        dsl = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo {.bindingCount = (uint32_t) binds.size(), .pBindings = binds.data()});
        vk::PushConstantRange pcr {.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(ConvPush)};
        pl = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo {.setLayoutCount = 1, .pSetLayouts = &*dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr});
        auto conv_sh = PyroWave::load_shader(device, "yuv2rgba");
        vk::PipelineShaderStageCreateInfo conv_st {.stage = vk::ShaderStageFlagBits::eCompute, .module = *conv_sh, .pName = "main"};
        pipe = vk::raii::Pipeline(device, nullptr, vk::ComputePipelineCreateInfo {.stage = conv_st, .layout = *pl});
        std::array<vk::DescriptorPoolSize, 3> psz {{
          {.type = vk::DescriptorType::eSampledImage, .descriptorCount = 3},
          {.type = vk::DescriptorType::eSampler,      .descriptorCount = 1},
          {.type = vk::DescriptorType::eStorageImage, .descriptorCount = 1},
        }};
        dpool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo {.maxSets = 1, .poolSizeCount = (uint32_t) psz.size(), .pPoolSizes = psz.data()});
        dset = std::move(vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo {.descriptorPool = *dpool, .descriptorSetCount = 1, .pSetLayouts = &*dsl}).front());
        vk::DescriptorImageInfo yi {.imageView = *view_y,  .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorImageInfo cbi {.imageView = *view_cb, .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorImageInfo cri {.imageView = *view_cr, .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorImageInfo si {.sampler = *sampler};
        vk::DescriptorImageInfo oi {.imageView = *rgba_view, .imageLayout = vk::ImageLayout::eGeneral};
        std::array<vk::WriteDescriptorSet, 5> ws {{
          {.dstSet = *dset, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &yi},
          {.dstSet = *dset, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &cbi},
          {.dstSet = *dset, .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &cri},
          {.dstSet = *dset, .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler,      .pImageInfo = &si},
          {.dstSet = *dset, .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &oi},
        }};
        device.updateDescriptorSets(ws, {});
        return true;
      } catch (const std::exception &e) {
        PWLOG(ANDROID_LOG_ERROR, "vk resource init failed: %s", e.what());
        return false;
      }
    }

    // (Re)creates the swapchain and derived objects for the existing surface.
    // Called from init() and again on out-of-date/suboptimal (resize, rotation,
    // compositor changes). Caller must ensure no GPU work is in flight when
    // recreating.
    bool create_swapchain() {
      try {
        auto &device = ctx->device();
        auto &phys = ctx->physical_device();

        auto caps = phys.getSurfaceCapabilitiesKHR(*surface);
        auto formats = phys.getSurfaceFormatsKHR(*surface);
        vk::ColorSpaceKHR colorspace = formats.empty() ? vk::ColorSpaceKHR::eSrgbNonlinear : formats[0].colorSpace;
        swap_format = formats.empty() ? vk::Format::eR8G8B8A8Unorm : formats[0].format;
        for (auto &f : formats) {
          if (f.format == vk::Format::eR8G8B8A8Unorm || f.format == vk::Format::eB8G8R8A8Unorm) {
            swap_format = f.format; colorspace = f.colorSpace; break;
          }
        }
        swap_extent = (caps.currentExtent.width != 0xFFFFFFFFu)
          ? caps.currentExtent : vk::Extent2D {(uint32_t) width, (uint32_t) height};
        uint32_t img_count = std::max(caps.minImageCount, 3u);
        if (caps.maxImageCount) img_count = std::min(img_count, caps.maxImageCount);

        // Direct present (skip the offscreen image + blit) is only safe when the
        // swapchain image can be a storage image AND its format is R8G8B8A8 - the
        // yuv2rgba shader stores rgba8 by component, so a B8G8R8A8 target would swap
        // R/B (the blit path does that conversion for us, the compute store does not).
        direct_present = (swap_format == vk::Format::eR8G8B8A8Unorm) &&
                         (bool) (caps.supportedUsageFlags & vk::ImageUsageFlagBits::eStorage);

        // Use IDENTITY pre-transform so we never render pre-rotated: on a phone the
        // surface's currentTransform is a 90/270 rotation (portrait-native panel,
        // landscape game), and applying it without rotating the content shows the
        // video sideways. Let the compositor do the rotation. currentExtent already
        // matches the presentation orientation.
        auto pretransform = (caps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity)
          ? vk::SurfaceTransformFlagBitsKHR::eIdentity : caps.currentTransform;

        // Prefer MAILBOX: tear-free and never blocks. With the host's capture
        // phase-locked to this display (display-timing feedback) frames arrive
        // just-in-time, so mailbox adds no queueing latency in steady state.
        // Then IMMEDIATE (tears), then FIFO (always available on Android).
        auto present_modes = phys.getSurfacePresentModesKHR(*surface);
        auto has_mode = [&](vk::PresentModeKHR m) {
          return std::find(present_modes.begin(), present_modes.end(), m) != present_modes.end();
        };
        vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
        if (has_mode(vk::PresentModeKHR::eMailbox)) present_mode = vk::PresentModeKHR::eMailbox;
        else if (has_mode(vk::PresentModeKHR::eImmediate)) present_mode = vk::PresentModeKHR::eImmediate;

        use_display_timing = ctx->caps().display_timing;

        swapchain = vk::raii::SwapchainKHR(device, vk::SwapchainCreateInfoKHR {
          .surface = *surface, .minImageCount = img_count,
          .imageFormat = swap_format, .imageColorSpace = colorspace,
          .imageExtent = swap_extent, .imageArrayLayers = 1,
          .imageUsage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment |
                        (direct_present ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits{}),
          .imageSharingMode = vk::SharingMode::eExclusive,
          .preTransform = pretransform,
          .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
          .presentMode = present_mode,
          .clipped = true});
        swap_images = swapchain.getImages();
        if (direct_present) {
          for (auto img : swap_images) {
            swap_storage_views.push_back(device.createImageView(vk::ImageViewCreateInfo {
              .image = img, .viewType = vk::ImageViewType::e2D, .format = swap_format,
              .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}}));
          }
        }
        PWLOG(ANDROID_LOG_INFO, "swapchain %ux%u present_mode=%d transform=%d direct_present=%d display_timing=%d",
              swap_extent.width, swap_extent.height, (int) present_mode, (int) pretransform, (int) direct_present,
              (int) use_display_timing);

        // Present IDs are per swapchain; reset the display-timing tracking.
        pending_present_times.clear();
        present_counter = 0;
        return true;
      } catch (const std::exception &e) {
        PWLOG(ANDROID_LOG_ERROR, "swapchain creation failed: %s", e.what());
        return false;
      }
    }

    bool recreate_swapchain() {
      PWLOG(ANDROID_LOG_INFO, "recreating swapchain (resize/rotation/out-of-date)");
      wait_prev();
      try { (void) ctx->device().waitIdle(); } catch (...) {}
      swap_storage_views.clear();
      swap_images.clear();
      swapchain = nullptr;
      swapchain_stale = false;
      first_frame_done = false;  // re-arm the pacing cold start
      return create_swapchain();
    }

    // Decode the current frame into the YCbCr planes, convert to RGBA on the GPU,
    // blit onto an acquired swapchain image, and present. No CPU readback.
    bool decode_and_present() {
      auto &device = ctx->device();

      // The previous frame saw out-of-date/suboptimal (rotation, resize,
      // compositor change): rebuild the swapchain before decoding into it.
      if (swapchain_stale && !recreate_swapchain()) {
        PWLOG(ANDROID_LOG_ERROR, "swapchain recreation failed");
        return false;
      }

      uint32_t idx = 0;
      auto acquire_start = std::chrono::steady_clock::now();
      try {
        auto [res, i] = swapchain.acquireNextImage(UINT64_MAX, *acquire_sem[parity], nullptr);
        if (res == vk::Result::eSuboptimalKHR) {
          // Usable this frame, but rebuild before the next one.
          swapchain_stale = true;
        } else if (res != vk::Result::eSuccess) {
          return true;  // transient; skip this frame
        }
        idx = i;
      } catch (const vk::OutOfDateKHRError &) {
        swapchain_stale = true;  // rebuilt at the start of the next frame
        return true;
      } catch (const vk::SystemError &e) {
        PWLOG(ANDROID_LOG_WARN, "acquireNextImage failed: %s", e.what());
        return true;
      }
      if (!first_frame_done) {
        first_frame_done = true;  // skip cold-start acquire (no steady state yet)
      } else if (!use_display_timing) {
        // Fallback signal (meaningful under FIFO only): time blocked in acquire.
        auto block_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - acquire_start).count();
        if (block_us < 0) block_us = 0;
        if (block_us > 50000) block_us = 50000;  // clamp pathological stalls (~50 ms)
        last_phase_offset_us = (int) block_us;
      }

      cmd.reset();
      cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

      auto barrier = [&](vk::Image image, vk::ImageLayout oldL, vk::ImageLayout newL,
                         vk::AccessFlags src, vk::AccessFlags dst,
                         vk::PipelineStageFlags ss, vk::PipelineStageFlags ds) {
        cmd.pipelineBarrier(ss, ds, {}, {}, {}, vk::ImageMemoryBarrier {
          .srcAccessMask = src, .dstAccessMask = dst, .oldLayout = oldL, .newLayout = newL,
          .image = image,
          .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}});
      };

      // Planes -> General for the decode fragment path (renders to them as attachments).
      auto plane_old = images_initialized ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined;
      for (vk::Image p : {vk::Image(img_y), vk::Image(img_cb), vk::Image(img_cr)}) {
        barrier(p, plane_old, vk::ImageLayout::eGeneral,
                images_initialized ? vk::AccessFlagBits::eShaderRead : vk::AccessFlags{},
                vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eShaderWrite,
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eComputeShader);
      }
      images_initialized = true;

      PyroWave::Decoder::ViewBuffers views {*view_y, *view_cb, *view_cr};
      if (!decoder->decode(cmd, *input, views)) {
        PWLOG(ANDROID_LOG_WARN, "decode() recorded failure");
      }

      // Planes: decode-write -> compute-sample.
      for (vk::Image p : {vk::Image(img_y), vk::Image(img_cb), vk::Image(img_cr)}) {
        barrier(p, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead,
                vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eComputeShader);
      }

      vk::PipelineStageFlags wait_stage;
      if (direct_present) {
        // Convert straight into the acquired swapchain image (the shader's UV sampling
        // scales decode-res planes -> surface-res output), skipping the offscreen image
        // and the blit copy. The dset's output binding is repointed at this image; the
        // previous frame's GPU work already finished (wait_prev) so the update is safe.
        vk::DescriptorImageInfo oi {.imageView = *swap_storage_views[idx], .imageLayout = vk::ImageLayout::eGeneral};
        vk::WriteDescriptorSet w {.dstSet = *dset, .dstBinding = 4, .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &oi};
        ctx->device().updateDescriptorSets(w, {});

        barrier(swap_images[idx], vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                {}, vk::AccessFlagBits::eShaderWrite,
                vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader);

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipe);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pl, 0, *dset, {});
        const bool scaling = swap_extent.width != uint32_t(width) || swap_extent.height != uint32_t(height);
        ConvPush push {(int32_t) swap_extent.width, (int32_t) swap_extent.height,
                       1.0f / float(swap_extent.width), 1.0f / float(swap_extent.height),
                       width, height, scaling ? 1 : 0};
        cmd.pushConstants<ConvPush>(*pl, vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch((swap_extent.width + 7) / 8, (swap_extent.height + 7) / 8, 1);

        barrier(swap_images[idx], vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR,
                vk::AccessFlagBits::eShaderWrite, {},
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eBottomOfPipe);

        wait_stage = vk::PipelineStageFlagBits::eComputeShader;
      } else {
        // Offscreen RGBA -> General for compute write (fully overwritten, discard old).
        barrier(img_rgba, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                {}, vk::AccessFlagBits::eShaderWrite,
                vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader);

        // Repoint the output binding at the offscreen image: a previous
        // direct-present frame (or a pre-recreation swapchain) may have left it
        // pointing at a swapchain view that no longer exists.
        vk::DescriptorImageInfo oi {.imageView = *rgba_view, .imageLayout = vk::ImageLayout::eGeneral};
        vk::WriteDescriptorSet w {.dstSet = *dset, .dstBinding = 4, .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &oi};
        ctx->device().updateDescriptorSets(w, {});

        // Dispatch yuv2rgba.
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipe);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pl, 0, *dset, {});
        ConvPush push {width, height, 1.0f / float(width), 1.0f / float(height),
                       width, height, 0};
        cmd.pushConstants<ConvPush>(*pl, vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.dispatch((uint32_t(width) + 7) / 8, (uint32_t(height) + 7) / 8, 1);

        // RGBA compute-write -> blit src.
        barrier(img_rgba, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead,
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer);
        // Swapchain image -> blit dst.
        barrier(swap_images[idx], vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                {}, vk::AccessFlagBits::eTransferWrite,
                vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer);

        // Blit decode-res RGBA -> surface-extent swapchain image (linear scale).
        vk::ImageBlit blit {
          .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
          .srcOffsets = std::array<vk::Offset3D, 2> {vk::Offset3D {0, 0, 0}, vk::Offset3D {width, height, 1}},
          .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
          .dstOffsets = std::array<vk::Offset3D, 2> {vk::Offset3D {0, 0, 0}, vk::Offset3D {(int32_t) swap_extent.width, (int32_t) swap_extent.height, 1}},
        };
        cmd.blitImage(img_rgba, vk::ImageLayout::eTransferSrcOptimal,
                      swap_images[idx], vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

        // Swapchain image -> present src.
        barrier(swap_images[idx], vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
                vk::AccessFlagBits::eTransferWrite, {},
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe);

        wait_stage = vk::PipelineStageFlagBits::eTransfer;
      }

      cmd.end();
      device.resetFences(*fence);
      ctx->queue().submit(vk::SubmitInfo {
        .waitSemaphoreCount = 1, .pWaitSemaphores = &*acquire_sem[parity], .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1, .pCommandBuffers = &*cmd,
        .signalSemaphoreCount = 1, .pSignalSemaphores = &*present_sem[parity]}, *fence);

      vk::SwapchainKHR sc = *swapchain;
      vk::PresentTimeGOOGLE ptime {};
      vk::PresentTimesInfoGOOGLE ptimes {};
      vk::PresentInfoKHR present_info {
        .waitSemaphoreCount = 1, .pWaitSemaphores = &*present_sem[parity],
        .swapchainCount = 1, .pSwapchains = &sc, .pImageIndices = &idx};
      if (use_display_timing) {
        ptime.presentID = ++present_counter;
        ptime.desiredPresentTime = 0;  // as soon as possible
        ptimes.swapchainCount = 1;
        ptimes.pTimes = &ptime;
        present_info.pNext = &ptimes;
      }
      try {
        (void) ctx->queue().presentKHR(present_info);
        if (use_display_timing) {
          pending_present_times.emplace_back(ptime.presentID, now_monotonic_ns());
          while (pending_present_times.size() > 32) {
            pending_present_times.pop_front();
          }
          poll_display_timing();
        }
      } catch (const vk::OutOfDateKHRError &) {
        swapchain_stale = true;  // rebuilt at the start of the next frame
      } catch (const vk::SystemError &e) {
        PWLOG(ANDROID_LOG_WARN, "presentKHR failed: %s", e.what());
      }

      // Pipelined: do NOT wait here. The next frame's wait_prev() waits this GPU
      // work, so the decode of this frame overlaps the network receive of the next.
      have_pending = true;
      parity ^= 1;
      return true;
    }
  };

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_limelight_binding_video_PyroWaveDecoder_nativeSetup(
    JNIEnv *env, jobject /*thiz*/, jobject surface, jint width, jint height) {
  auto ctx = pyrowave_vk::context::create();
  if (!ctx) {
    PWLOG(ANDROID_LOG_ERROR, "Vulkan context unavailable");
    return 0;
  }
  auto state = std::make_unique<decoder_state>();
  state->ctx = std::move(ctx);
  state->width = width;
  state->height = height;

  // Path selection follows upstream Decoder::device_prefers_fragment_path (commit 7c239af):
  // Qualcomm/Adreno and ARM/Mali strongly prefer the FRAGMENT IDWT path (compute is slow
  // and buggy on those drivers); NVIDIA (incl. Tegra X1), AMD and Intel use the COMPUTE
  // path. Our earlier logic (fragment = !is_adreno) had this inverted for BOTH classes -
  // Adreno was wrongly on compute and Tegra wrongly on fragment. Query driver_id when the
  // driver exposes it (1.2+ / VK_KHR_driver_properties); fall back to the device-name string
  // on 1.1 drivers (e.g. older Adreno, Tegra X1) where driver_id reads as 0/unknown.
  auto dev_props = state->ctx->physical_device().getProperties();
  bool prefer_fragment = false;
  {
    auto dp = state->ctx->physical_device()
                  .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDriverProperties>()
                  .get<vk::PhysicalDeviceDriverProperties>();
    switch (dp.driverID) {
      case vk::DriverId::eQualcommProprietary:
      case vk::DriverId::eArmProprietary:
        prefer_fragment = true;
        break;
      default:
        break;
    }
    const char *name = dev_props.deviceName.data();
    if (std::strstr(name, "Adreno") || std::strstr(name, "Mali") || std::strstr(name, "Turnip"))
      prefer_fragment = true;  // covers Turnip/PanVK (Mesa) + 1.1 drivers
  }
  bool fragment_path = prefer_fragment;

  try {
    state->decoder = std::make_unique<PyroWave::Decoder>(
      state->ctx->physical_device(), state->ctx->device(), width, height,
      PyroWave::ChromaSubsampling::Chroma420, fragment_path);
    {
      auto f12 = state->ctx->physical_device().getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>().get<vk::PhysicalDeviceVulkan12Features>();
      PWLOG(ANDROID_LOG_INFO, "PyroWave decoder GPU='%s' IDWT=%s PRECISION(compile)=%d precision(runtime)=%d shaderFloat16=%d -> idwt_%d%s",
            dev_props.deviceName.data(), fragment_path ? "fragment" : "compute",
            (int) PYROWAVE_PRECISION, PyroWave::Configuration::get().get_precision(),
            (int) f12.shaderFloat16, (int) PYROWAVE_PRECISION, f12.shaderFloat16 ? "_fp16" : "");
    }
    state->input = std::make_unique<PyroWave::DecoderInput>(*state->decoder);
  } catch (const std::exception &e) {
    PWLOG(ANDROID_LOG_ERROR, "decoder init failed: %s", e.what());
    return 0;
  }
  // The window must exist before init() (it creates the Vulkan surface/swapchain).
  state->window = ANativeWindow_fromSurface(env, surface);
  if (!state->window) {
    PWLOG(ANDROID_LOG_ERROR, "ANativeWindow_fromSurface returned null");
    return 0;
  }
  if (!state->init()) {
    return 0;
  }
  PWLOG(ANDROID_LOG_INFO, "PyroWave decoder ready %dx%d (swapchain present)", width, height);
  return reinterpret_cast<jlong>(state.release());
}

JNIEXPORT jint JNICALL
Java_com_limelight_binding_video_PyroWaveDecoder_nativeSubmit(
    JNIEnv *env, jobject /*thiz*/, jlong handle, jbyteArray data, jint length) {
  auto *state = reinterpret_cast<decoder_state *>(handle);
  if (!state) return DR_NEED_IDR;

  // Pipelining: wait for the previously submitted frame's GPU work to finish
  // (this overlapped with the current frame's network receive/depacketize), then
  // reset the input for this frame. clear() touches a host buffer the GPU reads,
  // so it must come after the wait.
  state->wait_prev();
  state->input->clear();

  // GetPrimitiveArrayCritical avoids the copy GetByteArrayElements typically
  // makes for a multi-KB frame on the per-frame hot path. push_data() only
  // does CPU memcpys (no JNI calls, no blocking), so the critical section is
  // short and safe.
  void *bytes = env->GetPrimitiveArrayCritical(data, nullptr);
  if (!bytes) {
    return DR_NEED_IDR;
  }
  bool pushed = state->input->push_data(
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(bytes), (size_t) length));
  env->ReleasePrimitiveArrayCritical(data, bytes, JNI_ABORT);

  if (!pushed) {
    return DR_NEED_IDR;
  }

  // Incomplete full (code-0) frames would flash zeros in the missing regions,
  // so hold the last good frame instead. Keep-previous frames (conditional
  // replenishment) are safe to decode with missing blocks - the decoder simply
  // retains the previous coefficients there. Throttle the log to avoid spam.
  if (!state->input->is_complete() && !state->input->keep_previous_frame()) {
    static int incomplete_count = 0;
    if ((++incomplete_count % 30) == 1) {
      PWLOG(ANDROID_LOG_WARN, "incomplete frame %d/%d blocks - holding last good frame (count=%d)",
            state->input->blocks_decoded(), state->input->blocks_expected(), incomplete_count);
    }
    return DR_OK;
  }

  // Make the CPU-written bitstream/offset buffers visible to the GPU before decode.
  // On non-coherent HOST_CACHED memory (Tegra X1) a missing flush makes the decode
  // read the previous frame's bitstream from device memory - looks fine when static,
  // garbage on motion. No-op on coherent memory.
  state->input->flush();

  // Treat each decode unit as a complete PyroWave frame (intra-only, self-contained).
  bool ok = state->decode_and_present();
  return ok ? DR_OK : DR_NEED_IDR;
}

// Latest phase-offset measurement (microseconds blocked acquiring a swapchain image).
// Java reads this after each submit and forwards it to the host via MoonBridge.sendPhaseOffset.
JNIEXPORT jint JNICALL
Java_com_limelight_binding_video_PyroWaveDecoder_nativeGetPhaseOffsetUs(
    JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {
  if (handle == 0) { return 0; }
  // Consume on read: display-timing measurements arrive a few frames late, and
  // re-sending a stale error every frame would keep pushing the host's
  // bang-bang controller in one direction. A consumed 0 falls inside the
  // controller's deadband and is inert.
  auto *state = reinterpret_cast<decoder_state *>(handle);
  int v = state->last_phase_offset_us;
  state->last_phase_offset_us = 0;
  return v;
}

JNIEXPORT void JNICALL
Java_com_limelight_binding_video_PyroWaveDecoder_nativeCleanup(
    JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {
  delete reinterpret_cast<decoder_state *>(handle);
}

}  // extern "C"
