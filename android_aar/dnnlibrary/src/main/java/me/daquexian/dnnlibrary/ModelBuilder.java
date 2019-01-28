package me.daquexian.dnnlibrary;

import android.content.res.AssetManager;

/**
 * Created by daquexian on 2017/11/12.
 * Java wrapper for ModelBuilder
 */

public class ModelBuilder {

    static {
        System.loadLibrary( "daq-jni");
    }

    public static final int PREFERENCE_LOW_POWER = 0;
    public static final int PREFERENCE_FAST_SINGLE_ANSWER = 1;
    public static final int PREFERENCE_SUSTAINED_SPEED = 2;
    private long nativeHandle;
    public ModelBuilder() {
        initHandle();
    }
    public native ModelBuilder readFile(AssetManager assetManager, String filename);
    public native ModelBuilder setOutput(String blobName);
    public native ModelBuilder allowFp16(boolean allowed);
    public native Model compile(int preference);
    public native void dispose();
    private native void initHandle();
    public void finalize() {
        dispose();
    }
}
