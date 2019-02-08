package me.daquexian.dnnlibrary;

/**
 * Created by daquexian on 2018.08.27.
 * Java wrapper for Model
 */

public class Model {

    static {
        System.loadLibrary( "daq-jni");
    }

    private long nativeHandle;
    public native float[] predict(float[] input);
    public native float[] predict(byte[] input);
    public native byte[] predict_quant8(float[] input);
    public native byte[] predict_quant8(byte[] input);
    public native void dispose();
    public void finalize() {
        dispose();
    }
}
