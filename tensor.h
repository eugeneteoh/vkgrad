#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    char* device;
} Tensor;

extern "C" {
    Tensor* create_tensor(float* data, int* shape, int ndim, char* device);
}

#endif /* TENSOR_H */