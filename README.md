# Aurora for Android

Aurora is a fork of [Moonlight for Android](https://github.com/moonlight-stream/moonlight-android)
— the open source Android client for [Sunshine](https://github.com/LizardByte/Sunshine) and NVIDIA
GameStream — extended with support for **PyroWave**, a GPU-only intra-frame wavelet codec decoded
entirely in Vulkan compute. Paired with a [Solarflare](https://github.com/Koloses/Solarflare) host,
PyroWave delivers very low, fixed-latency streaming that bypasses the device's MediaCodec video
decoder entirely.

Everything Moonlight for Android does still works: Aurora remains fully compatible with stock
Sunshine and GameStream hosts using H.264, HEVC, and AV1.

Aurora uses its own application ID (`com.aurora.client`), so it installs alongside official
Moonlight rather than replacing it, per upstream's request that forks change the ID.

## What the fork adds

- **PyroWave decoder** (vendored from the [WiVRn](https://github.com/WiVRn/WiVRn) fork of
  [Hans-Kristian Arntzen's PyroWave](https://github.com/Themaister/pyrowave)): JNI Vulkan compute
  decode with pipelined presentation.
- **Display-timing frame pacing**: `VK_GOOGLE_display_timing` feedback reported to the host for
  phase-locked frame delivery.
- **Loss resilience**: partial-frame salvage and per-packet parser resynchronization, plus
  self-initializing decode state that converges via the host's rolling refresh.
- **New settings**: force the PyroWave codec, and opt into adaptive FEC and adaptive bitrate.
- **Rebranding**: Aurora name and artwork.

PyroWave requires a Solarflare host; there is also an
[Aurora PC client](https://github.com/Koloses/aurora-qt).

## Building

- Install Android Studio and the Android NDK.
- Run `git submodule update --init --recursive` from within the repository.
- Create `local.properties` with an `ndk.dir=` property pointing at your NDK.
- The PyroWave decoder needs the Vulkan C++ headers and shader generation tools: pass
  `-PvulkanHeaders=/path/containing/vulkan/vulkan.hpp` to gradle (or copy the headers into
  `app/src/main/jni/pyrowave/external/`), and have `python