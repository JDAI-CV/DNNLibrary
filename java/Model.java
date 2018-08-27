package me.daquexian.dnnlibrary;

/**
 * Created by daquexian on 2018.08.27.
 * Java wrapper for Model
 */

public class Model {

    static {
        System.loadLibrary( "dnnlibrary");
    }

    private long nativeHandle;
    public static native void setOutput(String blobName);
    public static native float[] predict(float[] input);
}
