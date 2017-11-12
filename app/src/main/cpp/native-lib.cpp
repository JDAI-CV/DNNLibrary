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
#include <sstream>

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

template <typename T>
std::string to_string(T value);

ModelBuilder builder;

extern "C"
JNIEXPORT void
JNICALL
Java_me_daquexian_nnapiexample_MainActivity_testSpeedInit(
        JNIEnv *env,
        jobject /* this */,
        jobject javaAssetManager) {

    AAssetManager *mgrr = AAssetManager_fromJava(env, javaAssetManager);
    if (builder.init(mgrr) != ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Create model error");
    }

    // uint32_t input = builder.addInput(224, 224, 20);

    /*
    float weight[5 * 5 * 20];
    for (auto i = 0; i < 5 * 5 * 20; i++) {
        weight[i] = 10.f * i / (5 * 5 * 20) - 5.f;
    }
    uint32_t weightInd = builder.addWeightOrBiasFromBuffer(weight, vector<uint32_t>{1, 5, 5, 20});
    float bias[20];
    for (auto i = 0; i < 20; i++) {
        bias[i] = i;
    }
    uint32_t biasInd = builder.addWeightOrBiasFromBuffer(bias, vector<uint32_t>{20});
     */

    builder.readFromFile("resnet18");

    // builder.addIndexIntoOutput(builder.getBlobIndex("res3a_branch1"));
    // builder.addIndexIntoOutput(builder.getBlobIndex("prob"));
    builder.addIndexIntoOutput(builder.getBlobIndex("fc1000"));
    builder.addIndexIntoOutput(builder.getBlobIndex("prob"));

    // uint32_t dw = builder.addDepthWiseConv(input, 1, 1, 0, 0, 0, 0, 5, 5,
                                           // ModelBuilder::ACTIVATION_NONE, 20, 1, weightInd, biasInd);

    /* uint32_t input = builder.addInput(2, 2, 2);
    float t[2]{1.f, 2.f};
    uint32_t tensorInd = builder.addWeightOrBiasFromBuffer(t, vector<uint32_t>{2});
    uint32_t add = builder.addAddTensor(input, tensorInd);
    builder.addIndexIntoOutput(add);*/
    // uint32_t fc = builder.addFC(input, 10, ModelBuilder::ACTIVATION_NONE, weightInd, biasInd);
    // uint32_t softmax1 = builder.addSoftMax(fc, 1.f);
    // uint32_t softmax2 = builder.addSoftMax(softmax1, 1.f);

    // builder.addIndexIntoOutput(fc);

    // builder.readFromFile("testspeedmodel");
    // builder.addIndexIntoOutput(builder.getBlobIndex("pool2"));

    // builder.addIndexIntoOutput(builder.getBlobIndex("conv1"));

    int ret;
    if ((ret = builder.compile(ModelBuilder::PREFERENCE_SUSTAINED_SPEED)) !=
        ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Create model error, code: " + to_string(ret));
    }
}

extern "C"
JNIEXPORT void
JNICALL
Java_me_daquexian_nnapiexample_MainActivity_testSpeedRun(
        JNIEnv *env, jobject /* this */ ){

    Model model;
    builder.prepareForExecution(model);

    // LOGD("%d", model.execution == nullptr);

    float data[224 * 224 * 3];
    for (auto i = 0; i < 224 * 224 * 3; i++) {
        data[i] = 0.5;
    }

    builder.setInputBuffer(model, builder.getInputIndexes()[0], data, sizeof(data));

    float fc1000[1000];
    builder.setOutputBuffer(model, builder.getOutputIndexes()[0], fc1000, sizeof(fc1000));
    float prob[1000];
    builder.setOutputBuffer(model, builder.getOutputIndexes()[1], prob, sizeof(prob));
    // float out[28][28][128];
    // builder.setOutputBuffer(model, builder.getOutputIndexes()[0], out, sizeof(out));
    // float prob[1000];
    // builder.setOutputBuffer(model, builder.getOutputIndexes()[0], prob, sizeof(prob));
    // builder.setOutputBuffer(model, builder.getOutputIndexes()[0], fc, sizeof(fc));
    // LOGD("%d", model.execution == nullptr);
    auto begin = clock();
    model.predict();
    auto end = clock();
    LOGD("time: %f", 1.f * (end - begin) / CLOCKS_PER_SEC);

    LOGD("max: %d, %f", getMaxIndex(fc1000, LENGTH(fc1000)), fc1000[getMaxIndex(fc1000, LENGTH(fc1000))]);
    LOGD("max: %d, %f", getMaxIndex(prob, LENGTH(prob)), prob[getMaxIndex(prob, LENGTH(prob))]);

    for (int i = 0; i < 1000; i++) {
        LOGD("fc1000: %f", fc1000[i]);
    }

    for (int i = 0; i < 1000; i++) {
        LOGD("prob: %f", prob[i]);
    }

    /*
    for (int i = 0; i < 64; i++) {
        LOGD("conv1: %f", conv1[0][0][i]);
    }
    for (int i = 0; i < 64; i++) {
        LOGD("pool1: %f", out[0][0][i]);
    }
     */
}

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

    builder.readFromFile("nnmodel");

    builder.addIndexIntoOutput(builder.getBlobIndex("ip2"));
    builder.addIndexIntoOutput(builder.getBlobIndex("prob"));


    int ret;
    if ((ret = builder.compile(ModelBuilder::PREFERENCE_SUSTAINED_SPEED)) !=
            ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Create model error, code: " + to_string(ret));
    }
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

    Model model;
    builder.prepareForExecution(model);
    builder.setInputBuffer(model, builder.getInputIndexes()[0], data, static_cast<size_t>(len));

    float ip2[10];
    builder.setOutputBuffer(model, builder.getOutputIndexes()[0], ip2, sizeof(ip2));
    float prob[10];
    builder.setOutputBuffer(model, builder.getOutputIndexes()[1], prob, sizeof(prob));

    model.predict();

    for (auto value : ip2) {
        LOGD("ip2: %f", value);
    }

    for (auto value : prob) {
        LOGD("prob: %f", value);
    }

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

template<typename T>
string to_string(T value) {
    ostringstream os ;
    os << value ;
    return os.str() ;
}



#pragma clang diagnostic pop