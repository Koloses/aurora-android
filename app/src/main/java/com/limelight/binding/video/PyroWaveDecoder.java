package com.limelight.binding.video;

import android.view.Surface;

/**
 * JNI wrapper around the native PyroWave Vulkan decoder (libpyrowave-decoder.so).
 *
 * PyroWave is a Vulkan GPU wavelet codec that cannot be decoded by MediaCodec, so
 * decode + present happen entirely in native code. {@link MediaCodecDecoderRenderer}
 * delegates to this class when the negotiated video format is
 * {@code VIDEO_FORMAT_PYROWAVE}.
 */
public class PyroWaveDecoder {
    static {
        System.loadLibrary("pyrowave-decoder");
    }

    private long handle;

    /** Create the native decoder targeting the given Surface. Returns false on failure. */
    public boolean setup(Surface surface, int width, int height) {
        handle = nativeSetup(surface, width, height);
        return handle != 0;
    }

    /** Submit one reassembled frame. Returns MoonBridge.DR_OK (0) or DR_NEED_IDR (-1). */
    public int submitDecodeUnit(byte[] data, int length) {
        if (handle == 0) {
            return -1;
        }
        return nativeSubmit(handle, data, length);
    }

    public void cleanup() {
        if (handle != 0) {
            nativeCleanup(handle);
            handle = 0;
        }
    }

    private native long nativeSetup(Surface surface, int width, int height);
    private native int nativeSubmit(long handle, byte[] data, int length);
    private native void nativeCleanup(long handle);
}
