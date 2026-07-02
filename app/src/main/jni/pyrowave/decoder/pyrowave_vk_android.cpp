/**
 * @file pyrowave_vk_android.cpp
 * @brief Android Vulkan context for PyroWave. Mirrors the desktop context but
 *        logs via __android_log_print. Headless (no swapchain). See header.
 */
#include "pyrowave_vk_android.h"

#include <android/log.h>
#include <algorithm>
#include <cstring>
#include <vector>

#include <vk_mem_alloc.h>

#define PWLOG(prio, ...) __android_log_print(prio, "PyroWaveVk", __VA_ARGS__)

namespace pyrowave_vk {

  namespace {
    int select_physical_device(const std::vector<vk::raii::PhysicalDevice> &devices) {
      int fallback = -1;
      for (int i = 0; i < (int) devices.size(); ++i) {
        auto props = devices[i].getProperties2();
        if (fallback < 0 || props.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
          fallback = i;
        }
      }
      return fallback;
    }

    uint32_t find_compute_family(vk::raii::PhysicalDevice &dev) {
      auto families = dev.getQueueFamilyProperties();
      for (uint32_t i = 0; i < families.size(); ++i) {
        if (families[i].queueFlags & vk::QueueFlagBits::eCompute) {
          return i;
        }
      }
      return UINT32_MAX;
    }
  }  // namespace

  std::unique_ptr<context> context::create() {
    try {
      auto self = std::unique_ptr<context>(new context());

      // Use the highest instance version the loader offers (capped at 1.3). The
      // codec queries VkPhysicalDeviceVulkan13Properties for subgroup-size info,
      // which the driver only fills when the instance is >= 1.3. The DEVICE is
      // still created with structs valid on any version, so a 1.1 physical device
      // also works (it just won't pass the codec's subgroup check, which is
      // expected - the codec genuinely needs subgroup size control).
      uint32_t loader_version = self->ctx.enumerateInstanceVersion();
      uint32_t api_version = std::min(loader_version, (uint32_t) VK_API_VERSION_1_3);
      vk::ApplicationInfo app_info {
        .pApplicationName = "moonlight-pyrowave",
        .apiVersion = api_version,
      };
      // Instance extensions for presenting to the Android Surface via a swapchain.
      static const char *wanted_inst_exts[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
      };
      std::vector<const char *> inst_exts;
      {
        auto avail = self->ctx.enumerateInstanceExtensionProperties();
        for (auto *w : wanted_inst_exts) {
          for (auto &e : avail) {
            if (std::strcmp(e.extensionName, w) == 0) { inst_exts.push_back(w); break; }
          }
        }
      }
      vk::InstanceCreateInfo inst_info {
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = (uint32_t) inst_exts.size(),
        .ppEnabledExtensionNames = inst_exts.data(),
      };
      self->inst = vk::raii::Instance(self->ctx, inst_info);

      vk::raii::PhysicalDevices phys_devices(self->inst);
      if (phys_devices.empty()) {
        PWLOG(ANDROID_LOG_ERROR, "no Vulkan physical devices");
        return nullptr;
      }
      int idx = select_physical_device(phys_devices);
      if (idx < 0) {
        return nullptr;
      }
      self->phys_dev = std::move(phys_devices[idx]);

      uint32_t cqf = find_compute_family(self->phys_dev);
      if (cqf == UINT32_MAX) {
        PWLOG(ANDROID_LOG_ERROR, "no compute queue family");
        return nullptr;
      }
      self->caps_.compute_queue_family = cqf;

      // On a Vulkan 1.1 device the 1.2/1.3 feature aggregates are invalid in the
      // device-create chain, so enable the codec's needs via individual KHR/EXT
      // extensions. 16-bit storage and YCbCr conversion are core 1.1. Mirrors
      // WiVRn's "Lower Vulkan requirement to 1.1" device setup.
      static const char *wanted_exts[] = {
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        // The codec records vkCmdPipelineBarrier2 (Synchronization2). On a 1.1
        // device (e.g. Adreno) that is not core, so the extension must be enabled
        // or the function pointer is null -> crash in CommandBuffer::pipelineBarrier2.
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        // Present the decoded frame via a swapchain (GPU compositable) instead of a
        // CPU readback + ANativeWindow software post.
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        // Actual-present-time feedback for host capture phase locking.
        VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME,
      };
      std::vector<const char *> enabled_exts;
      {
        auto avail = self->phys_dev.enumerateDeviceExtensionProperties();
        for (auto *w : wanted_exts) {
          for (auto &e : avail) {
            if (std::strcmp(e.extensionName, w) == 0) { enabled_exts.push_back(w); break; }
          }
        }
      }
      auto supported = self->phys_dev.getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDevice8BitStorageFeatures,
        vk::PhysicalDeviceSubgroupSizeControlFeatures,
        vk::PhysicalDeviceShaderFloat16Int8Features,
        vk::PhysicalDeviceTimelineSemaphoreFeatures,
        vk::PhysicalDeviceSynchronization2Features>();
      auto &q11 = supported.get<vk::PhysicalDeviceVulkan11Features>();
      auto &q8 = supported.get<vk::PhysicalDevice8BitStorageFeatures>();
      auto &qsg = supported.get<vk::PhysicalDeviceSubgroupSizeControlFeatures>();
      auto &qf16 = supported.get<vk::PhysicalDeviceShaderFloat16Int8Features>();
      auto &qts = supported.get<vk::PhysicalDeviceTimelineSemaphoreFeatures>();
      auto &qsync = supported.get<vk::PhysicalDeviceSynchronization2Features>();

      self->caps_.shader_float16 = qf16.shaderFloat16;
      self->caps_.timeline_semaphore = qts.timelineSemaphore;
      self->caps_.ycbcr_conversion = q11.samplerYcbcrConversion;
      for (auto *e : enabled_exts) {
        if (std::strcmp(e, VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME) == 0) {
          self->caps_.display_timing = true;
        }
      }

      float prio = 1.0f;
      vk::DeviceQueueCreateInfo queue_info {
        .queueFamilyIndex = cqf, .queueCount = 1, .pQueuePriorities = &prio};

      vk::PhysicalDeviceFeatures base_features {};
      base_features.shaderStorageImageWriteWithoutFormat = true;
      base_features.shaderInt16 = supported.get<vk::PhysicalDeviceFeatures2>().features.shaderInt16;

      vk::StructureChain<
        vk::DeviceCreateInfo,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDevice8BitStorageFeatures,
        vk::PhysicalDeviceSubgroupSizeControlFeatures,
        vk::PhysicalDeviceShaderFloat16Int8Features,
        vk::PhysicalDeviceTimelineSemaphoreFeatures,
        vk::PhysicalDeviceSynchronization2Features>
        dev_chain;
      auto &dci = dev_chain.get<vk::DeviceCreateInfo>();
      dci.queueCreateInfoCount = 1;
      dci.pQueueCreateInfos = &queue_info;
      dci.enabledExtensionCount = (uint32_t) enabled_exts.size();
      dci.ppEnabledExtensionNames = enabled_exts.data();
      dci.pEnabledFeatures = &base_features;

      auto &e11 = dev_chain.get<vk::PhysicalDeviceVulkan11Features>();
      e11.samplerYcbcrConversion = q11.samplerYcbcrConversion;
      e11.storageBuffer16BitAccess = q11.storageBuffer16BitAccess;

      dev_chain.get<vk::PhysicalDevice8BitStorageFeatures>().storageBuffer8BitAccess = q8.storageBuffer8BitAccess;
      auto &esg = dev_chain.get<vk::PhysicalDeviceSubgroupSizeControlFeatures>();
      esg.subgroupSizeControl = qsg.subgroupSizeControl;
      esg.computeFullSubgroups = qsg.computeFullSubgroups;
      dev_chain.get<vk::PhysicalDeviceShaderFloat16Int8Features>().shaderFloat16 = qf16.shaderFloat16;
      dev_chain.get<vk::PhysicalDeviceTimelineSemaphoreFeatures>().timelineSemaphore = qts.timelineSemaphore;
      dev_chain.get<vk::PhysicalDeviceSynchronization2Features>().synchronization2 = qsync.synchronization2;

      // Link each feature struct based on whether the FEATURE is supported, not on
      // whether it is exposed as an extension. On newer (1.2/1.3) devices these
      // features are core and not enumerable as extensions, but the feature
      // structs (and their queries) are still valid. Extension NAMES were only
      // added to enabled_exts above when actually enumerable.
      if (!q8.storageBuffer8BitAccess)
        dev_chain.unlink<vk::PhysicalDevice8BitStorageFeatures>();
      if (!(qsg.subgroupSizeControl && qsg.computeFullSubgroups))
        dev_chain.unlink<vk::PhysicalDeviceSubgroupSizeControlFeatures>();
      if (!qf16.shaderFloat16)
        dev_chain.unlink<vk::PhysicalDeviceShaderFloat16Int8Features>();
      if (!qts.timelineSemaphore)
        dev_chain.unlink<vk::PhysicalDeviceTimelineSemaphoreFeatures>();
      if (!qsync.synchronization2)
        dev_chain.unlink<vk::PhysicalDeviceSynchronization2Features>();

      self->dev = vk::raii::Device(self->phys_dev, dev_chain.get<vk::DeviceCreateInfo>());
      self->compute_queue = vk::raii::Queue(self->dev, cqf, 0);

      VmaVulkanFunctions vma_fns {};
      vma_fns.vkGetInstanceProcAddr = self->ctx.getDispatcher()->vkGetInstanceProcAddr;
      vma_fns.vkGetDeviceProcAddr = self->dev.getDispatcher()->vkGetDeviceProcAddr;
      VmaAllocatorCreateInfo aci {};
      aci.physicalDevice = *self->phys_dev;
      aci.device = *self->dev;
      aci.instance = *self->inst;
      // VMA must use the version the DEVICE implements (it loads device functions
      // by that version). The instance may be 1.3 (for the codec's property query)
      // while the physical device is only 1.1 - using 1.3 here would make VMA try
      // to load 1.3-core functions (e.g. vkGetDeviceBufferMemoryRequirements) that
      // a 1.1 device does not expose, and assert.
      aci.vulkanApiVersion = std::min(api_version, self->phys_dev.getProperties().apiVersion);
      aci.pVulkanFunctions = &vma_fns;
      self->allocator.emplace(aci, /*has_debug_utils=*/false);

      PWLOG(ANDROID_LOG_INFO, "Vulkan context ready (fp16=%d)", (int) self->caps_.shader_float16);
      return self;
    } catch (const std::exception &e) {
      PWLOG(ANDROID_LOG_ERROR, "context create failed: %s", e.what());
      return nullptr;
    }
  }

  context::~context() {
    allocator.reset();
  }

}  // namespace pyrowave_vk
