//
// Created by daquexian on 2017/11/12.
//

#include <map>
#include <vector>

#include <android/asset_manager_jni.h>
#include <dnnlibrary/DaqReader.h>
#include <dnnlibrary/ModelBuilder.h>
#include "jni_handle.h"

using dnn::DaqReader;
using dnn::Model;
using dnn::ModelBuilder;
using std::map;
using std::string;

jint throwException(JNIEnv *env, std::string message);

extern "C" JNIEXPORT jobject JNICALL
Java_me_daquexian_dnnlibrary_ModelBuilder_readFile(JNIEnv *env,
                                                   jobject obj /* this */,
                                                   jobject javaAssetManager,
                                                   jstring javaFilename) {
    ModelBuilder *builder = getHandle<ModelBuilder>(env, obj);
    DaqReader daq_reader;

    string filename = string(env->GetStringUTFChars(javaFilename, nullptr));
    AAssetManager *mgrr = AAssetManager_fromJava(env, javaAssetManager);

    AAsset *asset =
        AAssetManager_open(mgrr, filename.c_str(), AASSET_MODE_UNKNOWN);
    const uint8_t *buf = static_cast<const uint8_t *>(AAsset_getBuffer(asset));
    daq_reader.ReadDaq(buf, *builder);
    return obj;
}

extern "C" JNIEXPORT jobject JNICALL
Java_me_daquexian_dnnlibrary_ModelBuilder_setOutput(JNIEnv *env,
                                                    jobject obj /* this */,
                                                    jstring javaBlobName) {
    ModelBuilder *builder = getHandle<ModelBuilder>(env, obj);
    string blobName = string(env->GetStringUTFChars(javaBlobName, nullptr));
    builder->AddOutput(blobName);
    return obj;
}

extern "C" JNIEXPORT jobject JNICALL
Java_me_daquexian_dnnlibrary_ModelBuilder_allowFp16(JNIEnv *env,
                                                    jobject obj /* this */,
                                                    jboolean allowed) {
    ModelBuilder *builder = getHandle<ModelBuilder>(env, obj);
    builder->AllowFp16(allowed);
    return obj;
}

extern "C" JNIEXPORT jobject JNICALL
Java_me_daquexian_dnnlibrary_ModelBuilder_compile(JNIEnv *env,
                                                  jobject obj /* this */,
                                                  jint preference) {
    ModelBuilder *builder = getHandle<ModelBuilder>(env, obj);
    auto model = builder->Compile(preference)
                     .release();  // release raw pointer from smart pointer, we
                                  // have to manage it ourselves
    jclass cls = env->FindClass("me/daquexian/dnnlibrary/Model");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "()V");
    jobject model_obj = env->NewObject(cls, ctor);
    setHandle(env, model_obj, model);
    return model_obj;
}

#define DEFINE_PREDICT(name, cpp_input_type, jni_input_type, JniInputType,     \
                       cpp_output_type, jni_output_type, JniOutputType)        \
    extern "C" JNIEXPORT jni_output_type##Array JNICALL                        \
        Java_me_daquexian_dnnlibrary_Model_##name(                             \
            JNIEnv *env, jobject obj /* this */,                               \
            jni_input_type##Array dataArrayObject) {                           \
        Model *model = getHandle<Model>(env, obj);                             \
                                                                               \
        jni_input_type *data =                                                 \
            env->Get##JniInputType##ArrayElements(dataArrayObject, nullptr);   \
                                                                               \
        uint32_t output_len = model->GetSize(model->GetOutputs()[0]);          \
        cpp_output_type output[output_len];                                    \
        model->SetOutputBuffer(0, output);                                     \
                                                                               \
        model->Predict(std::vector{reinterpret_cast<cpp_input_type *>(data)}); \
                                                                               \
        jni_output_type##Array result =                                        \
            env->New##JniOutputType##Array(output_len);                        \
        env->Set##JniOutputType##ArrayRegion(                                  \
            result, 0, output_len,                                             \
            reinterpret_cast<jni_output_type *>(output));                      \
                                                                               \
        return result;                                                         \
    }

DEFINE_PREDICT(predict_1float_1float, float, jfloat, Float, float, jfloat,
               Float);
DEFINE_PREDICT(predict_1float_1quant8, float, jfloat, Float, uint8_t, jbyte,
               Byte);
DEFINE_PREDICT(predict_1quant8_1float, uint8_t, jbyte, Byte, float, jfloat,
               Float);
DEFINE_PREDICT(predict_1quant8_1quant8, uint8_t, jbyte, Byte, uint8_t, jbyte,
               Byte);

extern "C" JNIEXPORT void JNICALL
Java_me_daquexian_dnnlibrary_ModelBuilder_initHandle(JNIEnv *env,
                                                     jobject obj /* this */) {
    ModelBuilder *builder = new ModelBuilder();
    setHandle(env, obj, builder);
}

extern "C" JNIEXPORT void JNICALL
Java_me_daquexian_dnnlibrary_ModelBuilder_dispose(JNIEnv *env,
                                                  jobject obj /* this */) {
    auto handle = getHandle<ModelBuilder>(env, obj);
    if (handle != nullptr) {
        delete handle;
        setHandle(env, obj, nullptr);
    }
}

extern "C" JNIEXPORT void JNICALL Java_me_daquexian_dnnlibrary_Model_dispose(
    JNIEnv *env, jobject obj /* this */) {
    auto handle = getHandle<Model>(env, obj);
    if (handle != nullptr) {
        delete handle;
        setHandle(env, obj, nullptr);
    }
}

jint throwException(JNIEnv *env, std::string message) {
    jclass exClass;
    std::string className = "java/lang/RuntimeException";

    exClass = env->FindClass(className.c_str());

    return env->ThrowNew(exClass, message.c_str());
}
