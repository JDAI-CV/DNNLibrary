//
// Created by daquexian on 8/3/18.
//

#ifndef PROJECT_NODE_H
#define PROJECT_NODE_H

#include <onnx/onnx_pb.h>
#include <string>

/**
 * Wrapping onnx::NodeProto for retrieving attribute values
 */
class NodeAttrHelper {
   public:
    NodeAttrHelper(ONNX_NAMESPACE::NodeProto proto);

    float get(const std::string &key, float def_val);
    int get(const std::string &key, int def_val);
    std::vector<float> get(const std::string &key, std::vector<float> def_val);
    std::vector<int> get(const std::string &key, std::vector<int> def_val);
    std::string get(const std::string &key, std::string def_val);

    bool has_attr(const std::string &key);

   private:
    ONNX_NAMESPACE::NodeProto node_;
};

#endif  // PROJECT_ATTRIBUTE_H
