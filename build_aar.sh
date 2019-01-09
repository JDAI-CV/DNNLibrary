#! /usr/bin/env bash

MY_ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
MY_ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$MY_ANDROID_HOME/ndk-bundle}"
mkdir -p build_jni && pushd build_jni
cmake -DCMAKE_SYSTEM_NAME=Android -DCMAKE_ANDROID_NDK=$MY_ANDROID_NDK_HOME -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a -DCMAKE_ANDROID_NDK_TOOLCHAIN_VERSION=clang -DCMAKE_ANDROID_STL_TYPE=c++_static -DCMAKE_SYSTEM_VERSION=28 -DDNN_BUILD_JNI=ON -DDNN_BUILD_SHARED_LIBS=ON -DDNN_BUILD_BIN=OFF ..
cmake --build . -- "-j$(nproc)"
popd
cp build_jni/dnnlibrary/libdnn-jni.so ./android_aar/dnnlibrary/src/main/jniLibs/arm64-v8a/libdaq-jni.so

# Increase version code and update version name
if (( $# >= 1 )); then
    NEW_VER_NUM=`sed -nE 's/versionCode ([0-9]+)/\1+1/p' android_aar/dnnlibrary/build.gradle | bc`
    sed -i -E "s/versionCode [0-9]+/versionCode $NEW_VER_NUM/" android_aar/dnnlibrary/build.gradle
    sed -i -E "s/versionName .+/versionName \"v$1\"/" android_aar/dnnlibrary/build.gradle
    sed -i -E "s/publishVersion = .+/publishVersion = \'$1\'/" android_aar/dnnlibrary/build.gradle
fi

pushd android_aar
ANDROID_HOME=$MY_ANDROID_HOME ./gradlew clean build

# Publishing is only for myself
if (( $# == 2 )); then
	echo "Publishing.."
	ANDROID_HOME=$MY_ANDROID_HOME ./gradlew bintrayUpload -PbintrayUser=daquexian566 -PbintrayKey=$2 -PdryRun=false
fi
popd
