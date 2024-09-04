# VKGRAD

Simple autograd engine with Vulkan.

```bash
g++ -g -shared -o libtensor.so -fPIC tensor.cpp cpu.cpp vulkan.cpp -lMoltenVK -std=c++17
glslangValidator -V add_tensor.comp -o add_tensor.spv
```

References:

https://towardsdatascience.com/recreating-pytorch-from-scratch-with-gpu-support-and-automatic-differentiation-8f565122a3cc