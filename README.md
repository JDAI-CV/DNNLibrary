# DNNLibrary

*Run neural network on your Android phone using the new NNAPI !*

Android 8.1 introduces Neural Networks API (NNAPI). Though it is in beta, it's very exciting to run a model in the "native" way supported by Android System. :)

DNNLirary is a wrapper of NNAPI. It lets you easily make the use of the new NNAPI introduced in Android 8.1. You can convert your caffemodel into `daq` format by the convert tool and run the model directly. 

The demo in this repo uses extracted weights of ResNet-18 to recongnize images(ResNet-18 branch), and also uses extracted weights of LeNet and recongnize a handwritten number(LeNet branch).

## Screenshot

This screenshot is from `ResNet-18` branch, which lets user pick an image instead of using camera.

![Screenshot1](screenshot_image_resnet.png)

This screenshot is from `LeNet` branch, which uses camera.

![Screenshot2](screenshot_camera_mnist.png)

## Introduction





## Preparation

Please make sure the Android System on your phone is 8.1+, or you may want to use API 27 emulator. The [beta version of NDK](https://developer.android.com/ndk/downloads/index.html#beta-downloads) is necessary for NNAPI. If you want to compile the demo please use Android Studio 3.0+.
