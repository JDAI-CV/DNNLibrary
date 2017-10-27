#include <jni.h>
#include <string>
#include <vector>
#include <android/NeuralNetworks.h>

using namespace std;

extern "C"
jint throwException( JNIEnv *env, char *message );
ANeuralNetworksOperandType getFloat32OperandTypeWithDims(std::vector<uint32_t> &dims);


JNIEXPORT jstring

JNICALL
Java_me_daquexian_nnapiexample_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";


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
    vector<uint32_t> conv2WeightsDims{50, 5, 5, 20};
    vector<uint32_t> ip1WeightsDims{500, 800};
    vector<uint32_t> ip2WeightsDims{800, 10};

    ANeuralNetworksOperandType dataType = getFloat32OperandTypeWithDims(dataDims);

    ANeuralNetworksOperandType conv1BlobType = getFloat32OperandTypeWithDims(conv1BlobDims);

    ANeuralNetworksOperandType pool1BlobType = getFloat32OperandTypeWithDims(pool1BlobDims);

    ANeuralNetworksOperandType conv2BlobType = getFloat32OperandTypeWithDims(conv2BlobDims);

    ANeuralNetworksOperandType pool2BlobType = getFloat32OperandTypeWithDims(pool2BlobDims);

    ANeuralNetworksOperandType ip1BlobType = getFloat32OperandTypeWithDims(ip1BlobDims);

    ANeuralNetworksOperandType ip2BlobType = getFloat32OperandTypeWithDims(ip2BlobDims);

    ANeuralNetworksOperandType conv1WeightsType = getFloat32OperandTypeWithDims(conv1WeightsDims);

    ANeuralNetworksOperandType conv2WeightsType = getFloat32OperandTypeWithDims(conv2WeightsDims);

    ANeuralNetworksOperandType ip1WeightsType = getFloat32OperandTypeWithDims(ip1WeightsDims);

    ANeuralNetworksOperandType ip2WeightsType = getFloat32OperandTypeWithDims(ip2WeightsDims);

    ANeuralNetworksOperandType strideOneType;
    strideOneType.type = ANEURALNETWORKS_INT32;
    strideOneType.scale = 0.f;
    strideOneType.zeroPoint = 0;
    strideOneType.dimensionCount = 0;
    strideOneType.dimensions = NULL;

    // Now we add the seven operands, in the same order defined in the diagram.
    ANeuralNetworksModel_addOperand(model, &dataType);  // operand 0
    ANeuralNetworksModel_addOperand(model, &conv1BlobType);  // operand 1
    ANeuralNetworksModel_addOperand(model, &pool1BlobType); // operand 2
    ANeuralNetworksModel_addOperand(model, &conv2BlobType);  // operand 3
    ANeuralNetworksModel_addOperand(model, &pool2BlobType);  // operand 4
    ANeuralNetworksModel_addOperand(model, &ip1BlobType); // operand 5
    ANeuralNetworksModel_addOperand(model, &ip2BlobType);  // operand 6
    ANeuralNetworksModel_addOperand(model, &conv1WeightsType);  // operand 7
    ANeuralNetworksModel_addOperand(model, &conv2WeightsType);  // operand 8
    ANeuralNetworksModel_addOperand(model, &ip1WeightsType);  // operand 9
    ANeuralNetworksModel_addOperand(model, &ip2WeightsType);  // operand 10
    ANeuralNetworksModel_addOperand(model, &strideOneType);  // operand 11

    return env->NewStringUTF(hello.c_str());

}

ANeuralNetworksOperandType getFloat32OperandTypeWithDims(std::vector<uint32_t> &dims) {
    ANeuralNetworksOperandType type;
    type.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    type.scale = 0.f;    // These fields are useful for quantized tensors.
    type.zeroPoint = 0;  // These fields are useful for quantized tensors.
    type.dimensionCount = static_cast<uint32_t>(dims.size());
    type.dimensions = &dims[0];
}

jint throwException(JNIEnv *env, std::string message) {
    jclass exClass;
    std::string className = "java/lang/RuntimeException" ;

    exClass = env->FindClass(className.c_str());

    return env->ThrowNew(exClass, message.c_str());
}
