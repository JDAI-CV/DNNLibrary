#include <jni.h>
#include <string>
#include <vector>
#include <numeric>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <android/log.h>
#include <string.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "ModelBuilder.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedMacroInspection"

using namespace std;

#define  LOG_TAG    "NNAPI Demo"

#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG,__VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)

#define LENGTH(x) sizeof((x)) / sizeof((x)[0])

jint throwException( JNIEnv *env, string message );

int getMaxIndex(float arr[], int length);

ANeuralNetworksCompilation* compilation = nullptr;

ModelBuilder builder;

extern "C"
JNIEXPORT void
JNICALL
Java_me_daquexian_nnapiexample_MainActivity_initModel(
        JNIEnv *env,
        jobject /* this */,
        jobject javaAssetManager) {

    AAssetManager *mgrr = AAssetManager_fromJava(env, javaAssetManager);
    if (builder.init(mgrr) != ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Create model error");
    }

    uint32_t data = builder.addInput(28, 28);
    uint32_t conv1 = builder.addConv("conv1", data, 1, 1, 0, 0, 5, 5, ModelBuilder::ACTIVATION_NONE, 20);
    uint32_t pool1 = builder.addPool(conv1, 2, 2, 0, 0, 2, 2, ModelBuilder::ACTIVATION_NONE,
                                     ModelBuilder::MAX_POOL);
    uint32_t conv2 = builder.addConv("conv2", pool1, 1, 1, 0, 0, 5, 5, ModelBuilder::ACTIVATION_NONE, 50);
    uint32_t pool2 = builder.addPool(conv2, 2, 2, 0, 0, 2, 2, ModelBuilder::ACTIVATION_NONE,
                                     ModelBuilder::MAX_POOL);
    uint32_t ip1 = builder.addFC("ip1", pool2, 500, ModelBuilder::ACTIVATION_RELU);
    uint32_t ip2 = builder.addFC("ip2", ip1, 10, ModelBuilder::ACTIVATION_NONE);

    builder.addIndexIntoOutput(ip2);
    builder.addIndexIntoOutput(conv1);

    int ret;
    if ((ret = builder.finish()) != ANEURALNETWORKS_NO_ERROR) {
        LOGD("ERROR!!!! %d", ret);
        return;
    }

    compilation = builder.compilation;
}


extern "C"
JNIEXPORT jint
JNICALL
Java_me_daquexian_nnapiexample_MainActivity_predict(
        JNIEnv *env,
        jobject /* this */,
        jfloatArray dataArrayObject) {
    jfloat *data = env->GetFloatArrayElements(dataArrayObject, nullptr);
    jsize len = env->GetArrayLength(dataArrayObject);

    ANeuralNetworksExecution* execution = nullptr;
    ANeuralNetworksExecution_create(compilation, &execution);

    ANeuralNetworksExecution_setInput(execution, 0, NULL, data, static_cast<size_t>(len));

    float prob[10];
    ANeuralNetworksExecution_setOutput(execution, 0, NULL, prob, sizeof(prob));

    float conv1[24][24][20];
    ANeuralNetworksExecution_setOutput(execution, 1, NULL, conv1, sizeof(conv1));

    ANeuralNetworksEvent* event = NULL;
    if (ANeuralNetworksExecution_startCompute(execution, &event) != ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Start running model error");
    }

    if (ANeuralNetworksEvent_wait(event) != ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Run model error");
    }

    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);

    return getMaxIndex(prob, LENGTH(prob));
}

extern "C"
JNIEXPORT void
JNICALL
Java_me_daquexian_nnapiexample_MainActivity_clearModel(
        JNIEnv *env,
        jobject /* this */) {
    builder.clear();
}

/**
 * set operand value from file in assets
 * @param model
 * @param mgr A pointer to AAssetManager got from Java's AssetManager
 * @param index The index of operand
 * @param filename The filename of weight or bias
 * @return a pointer to the buffer of weights or bias, according to the doc, the buffer should not
 * be modified until all executions complete, so please delete the buffer after the executions
 * complete.
 */
char* setOperandValueFromAssets(ANeuralNetworksModel *model, AAssetManager *mgr, int32_t index,
                                const char* filename) {
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_UNKNOWN);
    size_t size = static_cast<size_t>(AAsset_getLength(asset));
    char* buffer = new char[size];
    AAsset_read(asset, buffer, static_cast<size_t>(size));
    ANeuralNetworksModel_setOperandValue(model, index, buffer, size);
    return buffer;
}

ANeuralNetworksOperandType getFloat32OperandTypeWithDims(std::vector<uint32_t> &dims) {
    ANeuralNetworksOperandType type;
    type.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    type.scale = 0.f;
    type.zeroPoint = 0;
    type.dimensionCount = static_cast<uint32_t>(dims.size());
    type.dimensions = &dims[0];

    return type;
}

ANeuralNetworksOperandType getInt32OperandType() {
    ANeuralNetworksOperandType type;
    type.type = ANEURALNETWORKS_TENSOR_INT32;
    type.scale = 0.f;
    type.zeroPoint = 0;
    type.dimensionCount = 0;
    type.dimensions = nullptr;

    return type;
}

ANeuralNetworksOperandType getFloat32OperandType() {
    ANeuralNetworksOperandType type;
    type.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    type.scale = 0.f;
    type.zeroPoint = 0;
    type.dimensionCount = 0;
    type.dimensions = nullptr;

    return type;
}

int getMaxIndex(float arr[], int length) {
    int maxIndex = 0;
    auto max = arr[0];
    for (int i = 1; i < length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}


jint throwException(JNIEnv *env, std::string message) {
    jclass exClass;
    std::string className = "java/lang/RuntimeException" ;

    exClass = env->FindClass(className.c_str());

    return env->ThrowNew(exClass, message.c_str());
}


#pragma clang diagnostic pop