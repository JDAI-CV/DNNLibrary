#include <chrono>
#include <iostream>
#include <vector>

#include <common/helper.h>
#include <dnnlibrary/ModelBuilder.h>

using namespace android::nn::wrapper;
using dnn::ModelBuilder;

int main() {
    ModelBuilder builder;
    builder.Prepare();
    const auto devices = builder.GetDevices();
    if (devices.has_value()) {
        for (const auto &device : devices.value()) {
            PNT(device.name, device.feature_level, device.type, device.version);
        }
    } else {
        std::cout << "Cannot get devices" << std::endl;
    }
}

