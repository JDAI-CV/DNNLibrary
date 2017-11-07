#include <jni.h>
#include <string>
#include <vector>
#include <numeric>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <android/NeuralNetworks.h>
#include <android/log.h>
#include <string.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

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
ANeuralNetworksOperandType getFloat32OperandTypeWithDims(std::vector<uint32_t> &dims);

ANeuralNetworksOperandType getInt32OperandType();
ANeuralNetworksOperandType getFloat32OperandType();

int getMaxIndex(float arr[], int length);

char* setOperandValueFromAssets(ANeuralNetworksModel *model, AAssetManager *mgr, int32_t index,
                                const char* filename);

extern "C"
JNIEXPORT jint
JNICALL
Java_me_daquexian_nnapiexample_MainActivity_predict(
        JNIEnv *env,
        jobject /* this */,
        jobject javaAssetManager,
        jfloatArray dataArrayObject) {
    jfloat *data = env->GetFloatArrayElements(dataArrayObject, nullptr);
    jsize len = env->GetArrayLength(dataArrayObject);

    ANeuralNetworksModel* model = nullptr;
    if (ANeuralNetworksModel_create(&model) != ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Create model error");
    }

    vector<uint32_t> dataDims{1, 28, 28, 1};
    vector<uint32_t> conv1BlobDims{1, 24, 24, 20};
    vector<uint32_t> pool1BlobDims{1, 12, 12, 20};
    vector<uint32_t> conv2BlobDims{1, 8, 8, 50};
    vector<uint32_t> pool2BlobDims{1, 4, 4, 50};
    vector<uint32_t> ip1BlobDims{1, 800};
    vector<uint32_t> ip2BlobDims{1, 10};
    vector<uint32_t> probBlobDims{1, 10};

    vector<uint32_t> conv1WeightsDims{20, 5, 5, 1};
    vector<uint32_t> conv1BiasesDims{20};
    vector<uint32_t> conv2WeightsDims{50, 5, 5, 20};
    vector<uint32_t> conv2BiasesDims{50};
    vector<uint32_t> ip1WeightsDims{500, 800};
    vector<uint32_t> ip1BiasesDims{500};
    vector<uint32_t> ip2WeightsDims{10, 500};
    vector<uint32_t> ip2BiasesDims{10};

    ANeuralNetworksOperandType dataType = getFloat32OperandTypeWithDims(dataDims);
    ANeuralNetworksOperandType conv1BlobType = getFloat32OperandTypeWithDims(conv1BlobDims);
    ANeuralNetworksOperandType pool1BlobType = getFloat32OperandTypeWithDims(pool1BlobDims);
    ANeuralNetworksOperandType conv2BlobType = getFloat32OperandTypeWithDims(conv2BlobDims);
    ANeuralNetworksOperandType pool2BlobType = getFloat32OperandTypeWithDims(pool2BlobDims);
    ANeuralNetworksOperandType ip1BlobType = getFloat32OperandTypeWithDims(ip1BlobDims);
    ANeuralNetworksOperandType ip2BlobType = getFloat32OperandTypeWithDims(ip2BlobDims);
    ANeuralNetworksOperandType probBlobType = getFloat32OperandTypeWithDims(probBlobDims);

    ANeuralNetworksOperandType conv1WeightsType = getFloat32OperandTypeWithDims(conv1WeightsDims);
    ANeuralNetworksOperandType conv1BiasesType = getFloat32OperandTypeWithDims(conv1BiasesDims);
    ANeuralNetworksOperandType conv2WeightsType = getFloat32OperandTypeWithDims(conv2WeightsDims);
    ANeuralNetworksOperandType conv2BiasesType = getFloat32OperandTypeWithDims(conv2BiasesDims);
    ANeuralNetworksOperandType ip1WeightsType = getFloat32OperandTypeWithDims(ip1WeightsDims);
    ANeuralNetworksOperandType ip1BiasesType = getFloat32OperandTypeWithDims(ip1BiasesDims);
    ANeuralNetworksOperandType ip2WeightsType = getFloat32OperandTypeWithDims(ip2WeightsDims);
    ANeuralNetworksOperandType ip2BiasesType = getFloat32OperandTypeWithDims(ip2BiasesDims);

    ANeuralNetworksOperandType int32Type = getInt32OperandType();
    ANeuralNetworksOperandType float32Type = getFloat32OperandType();

    // Now we add the seven operands, in the same order defined in the diagram.
    ANeuralNetworksModel_addOperand(model, &dataType);  // operand 0
    ANeuralNetworksModel_addOperand(model, &conv1BlobType);  // operand 1
    ANeuralNetworksModel_addOperand(model, &pool1BlobType); // operand 2
    ANeuralNetworksModel_addOperand(model, &conv2BlobType);  // operand 3
    ANeuralNetworksModel_addOperand(model, &pool2BlobType);  // operand 4
    ANeuralNetworksModel_addOperand(model, &ip1BlobType); // operand 5
    ANeuralNetworksModel_addOperand(model, &ip2BlobType);  // operand 6
    ANeuralNetworksModel_addOperand(model, &conv1WeightsType);  // operand 7
    ANeuralNetworksModel_addOperand(model, &conv1BiasesType);  // operand 8
    ANeuralNetworksModel_addOperand(model, &conv2WeightsType);  // operand 9
    ANeuralNetworksModel_addOperand(model, &conv2BiasesType);  // operand 10
    ANeuralNetworksModel_addOperand(model, &ip1WeightsType);  // operand 11
    ANeuralNetworksModel_addOperand(model, &ip1BiasesType);  // operand 12
    ANeuralNetworksModel_addOperand(model, &ip2WeightsType);  // operand 13
    ANeuralNetworksModel_addOperand(model, &ip2BiasesType);  // operand 14
    ANeuralNetworksModel_addOperand(model, &int32Type);  // operand 15, int one
    ANeuralNetworksModel_addOperand(model, &int32Type);  // operand 16, valid padding
    ANeuralNetworksModel_addOperand(model, &int32Type);  // operand 17, none activation
    ANeuralNetworksModel_addOperand(model, &int32Type);  // operand 18, relu activation
    ANeuralNetworksModel_addOperand(model, &int32Type);  // operand 19, int two
    ANeuralNetworksModel_addOperand(model, &probBlobType);  // operand 20, prob
    ANeuralNetworksModel_addOperand(model, &float32Type);  // operand 21, float one

    AAssetManager *mgr = AAssetManager_fromJava(env, javaAssetManager);

    vector<char*> bufferPointers;
    bufferPointers.push_back(setOperandValueFromAssets(model, mgr, 7, "weights_and_biases/conv1_weights"));
    bufferPointers.push_back(setOperandValueFromAssets(model, mgr, 8, "weights_and_biases/conv1_biases"));
    bufferPointers.push_back(setOperandValueFromAssets(model, mgr, 9, "weights_and_biases/conv2_weights"));
    bufferPointers.push_back(setOperandValueFromAssets(model, mgr, 10, "weights_and_biases/conv2_biases"));
    bufferPointers.push_back(setOperandValueFromAssets(model, mgr, 11, "weights_and_biases/ip1_weights"));
    bufferPointers.push_back(setOperandValueFromAssets(model, mgr, 12, "weights_and_biases/ip1_biases"));
    bufferPointers.push_back(setOperandValueFromAssets(model, mgr, 13, "weights_and_biases/ip2_weights"));
    bufferPointers.push_back(setOperandValueFromAssets(model, mgr, 14, "weights_and_biases/ip2_biases"));

    int32_t intOne = 1;
    ANeuralNetworksModel_setOperandValue(model, 15, &intOne, sizeof(intOne));

    int32_t validPadding = ANEURALNETWORKS_PADDING_VALID;
    ANeuralNetworksModel_setOperandValue(model, 16, &validPadding, sizeof(validPadding));

    int32_t noneActivation = ANEURALNETWORKS_FUSED_NONE;
    ANeuralNetworksModel_setOperandValue(model, 17, &noneActivation, sizeof(noneActivation));

    int32_t reluActivation = ANEURALNETWORKS_FUSED_RELU;
    ANeuralNetworksModel_setOperandValue(model, 18, &reluActivation, sizeof(reluActivation));

    int32_t two = 2;
    ANeuralNetworksModel_setOperandValue(model, 19, &two, sizeof(two));

    float floatOne = 1;
    ANeuralNetworksModel_setOperandValue(model, 21, &floatOne, sizeof(floatOne));

    uint32_t conv1InputIndexes[] = {0, 7, 8, 16, 15, 15, 17};
    uint32_t conv1OutputIndexes[] = {1};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 7, conv1InputIndexes, 1, conv1OutputIndexes);

    uint32_t pool1InputIndexes[] = {1, 16, 19, 19, 19, 19, 17};
    uint32_t pool1OutputIndexes[] = {2};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MAX_POOL_2D, 7, pool1InputIndexes, 1, pool1OutputIndexes);

    uint32_t conv2InputIndexes[] = {2, 9, 10, 16, 15, 15, 17};
    uint32_t conv2OutputIndexes[] = {3};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 7, conv2InputIndexes, 1, conv2OutputIndexes);

    uint32_t pool2InputIndexes[] = {3, 16, 19, 19, 19, 19, 17};
    uint32_t pool2OutputIndexes[] = {4};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MAX_POOL_2D, 7, pool2InputIndexes, 1, pool2OutputIndexes);

    uint32_t ip1InputIndexes[] = {4, 11, 12, 18};
    uint32_t ip1OutputIndexes[] = {5};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED, 4, ip1InputIndexes, 1, ip1OutputIndexes);

    uint32_t ip2InputIndexes[] = {5, 13, 14, 17};
    uint32_t ip2OutputIndexes[] = {6};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED, 4, ip2InputIndexes, 1, ip2OutputIndexes);

    uint32_t probInputIndexes[] = {6, 21};
    uint32_t probOutputIndexes[] = {20};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_SOFTMAX, 2, probInputIndexes, 1, probOutputIndexes);

    uint32_t modelInputIndexes[] = {0};
    uint32_t modelOutputIndexes[] = {20};
    ANeuralNetworksModel_identifyInputsAndOutputs(model, LENGTH(modelInputIndexes), modelInputIndexes, LENGTH(modelOutputIndexes), modelOutputIndexes);

    if (ANeuralNetworksModel_finish(model) != ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Finish model error");
    }

    ANeuralNetworksCompilation* compilation;
    ANeuralNetworksCompilation_create(model, &compilation);

    ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);

    ANeuralNetworksCompilation_finish(compilation);

    ANeuralNetworksExecution* execution = nullptr;
    ANeuralNetworksExecution_create(compilation, &execution);

    ANeuralNetworksExecution_setInput(execution, 0, NULL, data, static_cast<size_t>(len));

    float prob[10];
    ANeuralNetworksExecution_setOutput(execution, 0, NULL, prob, sizeof(prob));

    ANeuralNetworksEvent* event = NULL;
    if (ANeuralNetworksExecution_startCompute(execution, &event) != ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Start running model error");
    }

    if (ANeuralNetworksEvent_wait(event) != ANEURALNETWORKS_NO_ERROR) {
        throwException(env, "Run model error");
    }

    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);

    // Cleanup
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);
    for (auto pointer : bufferPointers) {
        delete [] pointer;
    }

    for (auto value : prob) {
        LOGD("prob: %f", value);
    }

    return getMaxIndex(prob, LENGTH(prob));
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