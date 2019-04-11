#! /usr/bin/env bash
# It is used for travis ci

set -e

MY_ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
echo "Publishing.."
pushd android_aar
ANDROID_HOME=$MY_ANDROID_HOME ./gradlew bintrayUpload -PbintrayUser=daquexian566 -PbintrayKey=$1 -PdryRun=false
popd
