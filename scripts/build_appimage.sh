wget https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
wget https://github.com/linuxdeploy/linuxdeploy-plugin-appimage/releases/download/continuous/linuxdeploy-plugin-appimage-x86_64.AppImage

chmod +x linuxdeploy-*.AppImage
mkdir -p appimage/appdir/usr/bin
cp build_onnx2daq/tools/onnx2daq/onnx2daq appimage/appdir/usr/bin/
./linuxdeploy-x86_64.AppImage --appdir appimage/appdir -d appimage/onnx2daq.desktop -i appimage/onnx2daq.png --output appimage
mv `ls onnx2daq-*.AppImage` onnx2daq.AppImage
