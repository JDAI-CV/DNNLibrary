# DNNLibrary

*Run neural network on your Android phone using the new NNAPI !*

Android 8.1 introduces Neural Networks API (NNAPI). It's very exciting to run a model in the "native" way supported by Android System. :)

DNNLirary is a wrapper of NNAPI. It lets you easily make the use of the new NNAPI introduced in Android 8.1. You can convert your caffemodel into `daq` format by the [convert tool](https://github.com/daquexian/DNN_convert_tool) and run the model directly. 

The demo consists `daq` model files for LeNet, ResNet-18 and SqueezeNet, and you can get your own model conveniently from pretrained caffemodel.

For how to use this lib directly in your project, check out [Usage](#usage) (it's at the bottom)

## Screenshot

This screenshot is ResNet-18

![Screenshot image resnet](screenshot_image_resnet.png)

This screenshot is LeNet

![Screenshot camera mnist](screenshot_camera_mnist.png)

## Introduction

Android 8.1 introduces NNAPI. From my experient it is very efficient on my Pixel. For example, it takes [caffe-android-lib](https://github.com/sh1r0/caffe-android-lib) an average time of 43.42ms to do a convolution with 20 5\*5 filters on 224\*224 image, but it only takes 15.45ms for NNAPI -- about 1/3 of caffe-android-lib.

What's more, we can believe [depthwise convolution](https://arxiv.org/abs/1704.04861), which is useful on mobile devices, is optimized in NNAPI. It takes caffe-android-lib and NNAPI 82.32ms and 16.93ms respectively to do 5 * 5 depthwise conv on 224 \* 224 \* 20 input.

However, NNAPI is not friendly to normal Android developers. It is not designed to be used by normal developers directly. So I wrapped it into a library.

With DNNLibrary it's extremely easy to deploy your caffe model on Android 8.1+ phone. Here is my code to deploy the ResNet-18 on phone:

```
ModelWrapper.readFile(getAssets(), "resnet18");
ModelWrapper.setOutput("prob");
ModelWrapper.compile(ModelWrapper.PREFERENCE_FAST_SINGLE_ANSWER);

float[] result = ModelWrapper.predict(inputData);
```

Only four lines! And the model file is got from my [convert tool](https://github.com/daquexian/DNN_convert_tool) from pretrained caffemodel.

If you use the "raw" NNAPI, the code will increase dramatically. Setting up a LeNet needs 200+ lines. (For the 200+ lines LeNet you can check out the second commit of this repo)

## Usage

Please download dnnlibrary-release.aar in lastest [Release](https://github.com/daquexian/DNNLibrary/releases) in this repo and [import it in your project](https://developer.android.com/studio/projects/android-library.html#AddDependency). This library will also be published on jcenter soon. :)

## Preparation

Please make sure the Android System on your phone is 8.1+, or you may want to use API 27 emulator. The [latest version of NDK](https://developer.android.com/ndk/downloads/index.html) is necessary for NNAPI. If you want to compile the demo please use Android Studio 3.0+.
