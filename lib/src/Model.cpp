//
// Created by daquexian on 2017/11/8.
//

#include "Model.h"

Model::Model() {

}

Model::Model(ANeuralNetworksExecution *execution) :execution(execution){

}

int Model::predict() {
    ANeuralNetworksEvent* event = NULL;
    int ret;
    if ((ret = ANeuralNetworksExecution_startCompute(execution, &event)) != ANEURALNETWORKS_NO_ERROR) {
        return ret;
    }

    if ((ret = ANeuralNetworksEvent_wait(event)) != ANEURALNETWORKS_NO_ERROR) {
        return ret;
    }

    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);

    return 0;
}

