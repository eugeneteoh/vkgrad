#ifndef TENSOR_H
#define TENSOR_H

#include <vulkan/vulkan.h>

typedef struct {
    float* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    char* device;

    // vulkan
    VkBuffer buffer;
    VkDeviceMemory memory;
} Tensor;

extern "C" {
    Tensor* create_tensor(float* data, int* shape, int ndim, char* device);
    float get_item(Tensor* tensor, int* indices);
    void to_device(Tensor* tensor, char* target_device);
    Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* sub_tensor(Tensor* tensor1, Tensor* tensor2);
}

#endif /* TENSOR_H */