/**
 * @file pyrowave_vk_android.h
 * @brief Headless Vulkan context for the PyroWave decoder on Android.
 *
 * Same role as the desktop pyrowave_vk::context: owns a Vulkan instance/device
 * plus the VMA vk_allocator singleton the codec requires. No swapchain - the
 * decoder presents to the Android Surface via ANativeWindow CPU readback.
 *
 * NOT compiled/tested here (no Android NDK/Vulkan). Build with ndk-build.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <vulkan/vulkan_raii.hpp>
#include "vk/vk_allocator.h"

namespace pyrowave_vk {

  struct device_caps_t {
    bool shader_float16 = false;
    bool timeline_semaphore = false;
    bool ycbcr_conversion = false;
    // VK_GOOGLE_display_timing: lets the decoder learn each frame's actual
    // present time and phase-lock the host capture clock to this display
    // (smoothness without a jitter buffer). Widely available on Android.
    bool display_timing = false;
    uint32_t compute_queue_family = UINT32_MAX;
  };

  class context {
  public:
    static std::unique_ptr<context> create();
    ~context();

    context(const context &) = delete;
    context &operator=(const context &) = delete;

    vk::raii::Instance &instance() { return inst; }
    vk::raii::PhysicalDevice &physical_device() { return phys_dev; }
    vk::raii::Device &device() { return dev; }
    const device_caps_t &caps() const { return caps_; }
    vk::raii::Queue &queue() { return compute_queue; }

  private:
    context() = default;

    vk::raii::Context ctx {};
    vk::raii::Instance inst = nullptr;
    vk::raii::PhysicalDevice phys_dev = nullptr;
    vk::raii::Device dev = nullptr;
    vk::raii::Queue compute_queue = nullptr;
    device_caps_t caps_;
    std::optional<vk_allocator> allocator;
  };

}  // namespace pyrowave_vk
