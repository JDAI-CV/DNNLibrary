# DNNLibrary

*Run neural network on your Android phone using the new NNAPI !*

Android 8.1 introduces Neural Networks API (NNAPI). Though it is in beta, it's very exciting to run a model in the "native" way supported by Android System. :)

DNNLirary is a wrapper of NNAPI. It lets you easily make the use of the new NNAPI introduced in Android 8.1. You can convert your caffemodel into `daq` format by the [convert tool](https://github.com/daquexian/DNN_convert_tool) and run the model directly. 

The demo in this repo uses extracted weights of ResNet-18 to recongnize images(ResNet-18 branch), and also uses extracted weights of LeNet and recongnize a handwritten number(LeNet branch).

## Screenshot

This screenshot is from `ResNet-18` branch, which lets user pick an image instead of using camera.

![Screenshot image resnet](screenshot_image_resnet.png)

This screenshot is from `LeNet` branch, which uses camera.

![Screenshot camera mnist](screenshot_camera_mnist.png)

## Introduction

With DNNLibrary it's extremely easy to deploy your caffe model on Android 8.1+ phone. Here is my code to deploy the ResNet-18 on phone:

```
ModelWrapper.readFile(getAssets(), "resnet18");
ModelWrapper.setOutput("prob");
ModelWrapper.compile(ModelWrapper.PREFERENCE_FAST_SINGLE_ANSWER);

float[] result = ModelWrapper.predict(inputData);
```

Only four lines! And the model file is got from my [convert tool](https://github.com/daquexian/DNN_convert_tool) from pretrained caffemodel.

If you use the "raw" NNAPI, the code will increase dramatically. Setting up a LeNet need 200+ lines. (You may want to check out the first commit of this repo)

## Preparation

Please make sure the Android System on your phone is 8.1+, or you may want to use API 27 emulator. The [beta version of NDK](https://developer.android.com/ndk/downloads/index.html#beta-downloads) is necessary for NNAPI. If you want to compile the demo please use Android Studio 3.0+.
