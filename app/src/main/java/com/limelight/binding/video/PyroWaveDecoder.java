package com.limelight.binding.video;

import android.view.Surface;

import com.limelight.nvstream.jni.MoonBridge;

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
        int result = nativeSubmit(handle, data, length);
        // Phase-offset pacing: forward the native decoder's display-backpressure measurement
        // to the host so it can lock its capture cadence to this client's display. Inert on
        // non-Sunshine hosts (MoonBridge.sendPhaseOffset -> LiSendPhaseOffset early-returns).
        MoonBridge.sendPhaseOffset(nativeGetPhaseOffsetUs(handle));
        return result;
    }

    public void cleanup() {
        if (handle != 0) {
            nativeCleanup(handle);
            handle = 0;
        }
    }

    private native long nativeSetup(Surface surface, int width, int height);
    private native int nativeSubmit(long handle, byte[] data, int length);
    private native int nativeGetPhaseOffsetUs(long handle);
    private native void nativeCleanup(long handle);
}
