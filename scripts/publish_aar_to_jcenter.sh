#! /usr/bin/env bash
# It is used for travis ci

set -e

MY_ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
echo "Publishing.."
ANDROID_HOME=$MY_ANDROID_HOME ./gradlew bintrayUpload -PbintrayUser=daquexian566 -PbintrayKey=$BINTRAY_KEY -PdryRun=false
