package me.daquexian.dnnlibrary;

/**
 * Created by daquexian on 2018.08.27.
 * Java wrapper for Model
 */

public class Model {

    static {
        System.loadLibrary( "daq-jni");
    }

    public float[] predict(float[] input) {
        return predict_float_float(input);
    }

    public float[] predict(byte[] input) {
        return predict_quant8_float(input);
    }

    public byte[] predictQuant8(float[] input) {
        return predict_float_quant8(input);
    }

    public byte[] predictQuant8(byte[] input) {
        return predict_quant8_quant8(input);
    }

    private long nativeHandle;
    private native float[] predict_float_float(float[] input);
    private native float[] predict_quant8_float(byte[] input);
    private native byte[] predict_float_quant8(float[] input);
    private native byte[] predict_quant8_quant8(byte[] input);
    public native void dispose();
    public void finalize() {
        dispose();
    }
}
