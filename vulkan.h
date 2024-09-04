#ifndef VULKAN_H
#define VULKAN_H

#include "tensor.h"

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue queue;
    VkCommandPool commandPool;
} VulkanContext;

// Function declarations
VulkanContext* getVulkanContext();  // Returns a pointer to the global Vulkan context
void cpu_to_vulkan(Tensor* tensor);
void vulkan_to_cpu(Tensor* tensor);
void cleanup_tensor_vulkan(Tensor* tensor, VulkanContext* context);

// Helper function declarations
VkResult createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
void copyBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
VkInstance createInstance();
VkPhysicalDevice pickPhysicalDevice(VkInstance instance);
VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, VkQueue* queue);
VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex);

#endif /* VULKAN_H */