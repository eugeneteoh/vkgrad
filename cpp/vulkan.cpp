#include "vulkan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Singleton function to initialize and return the Vulkan context
VulkanContext* getVulkanContext() {
    static VulkanContext context;
    static int initialized = 0;

    if (!initialized) {
        context.instance = createInstance();
        context.physicalDevice = pickPhysicalDevice(context.instance);
        context.device = createLogicalDevice(context.physicalDevice, &context.queue);
        context.commandPool = createCommandPool(context.device, 0);  // Assuming queueFamilyIndex is 0
        context.descriptorPool = createDescriptorPool(context.device);  // Create descriptor pool
        initialized = 1;
    }

    return &context;
}


// Transfer tensor data from CPU to Vulkan
void cpu_to_vulkan(Tensor* tensor) {
    VulkanContext* context = getVulkanContext();

    // Step 1: Create the Vulkan buffer for the tensor data
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = tensor->size * sizeof(float);  // Total size in bytes
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(context->device, &bufferInfo, nullptr, &tensor->buffer) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create Vulkan buffer\n");
        exit(1);
    }

    // Step 2: Allocate memory for the Vulkan buffer
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(context->device, tensor->buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(context->physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(context->device, &allocInfo, nullptr, &tensor->memory) != VK_SUCCESS) {
        fprintf(stderr, "Failed to allocate Vulkan memory\n");
        exit(1);
    }

    // Step 3: Bind the buffer to the allocated memory
    vkBindBufferMemory(context->device, tensor->buffer, tensor->memory, 0);

    // Step 4: Create a staging buffer for data transfer (host-visible memory)
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(context->device, context->physicalDevice, tensor->size * sizeof(float),
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                 stagingBuffer, stagingBufferMemory);

    // Step 5: Copy the CPU data to the staging buffer
    void* data;
    vkMapMemory(context->device, stagingBufferMemory, 0, tensor->size * sizeof(float), 0, &data);
    memcpy(data, tensor->data, tensor->size * sizeof(float));  // Copy CPU data to staging buffer
    vkUnmapMemory(context->device, stagingBufferMemory);

    // Step 6: Copy the data from the staging buffer to the Vulkan buffer (GPU memory)
    copyBuffer(context->device, context->commandPool, context->queue, stagingBuffer, tensor->buffer, tensor->size * sizeof(float));

    // Step 7: Clean up the staging buffer
    vkDestroyBuffer(context->device, stagingBuffer, nullptr);
    vkFreeMemory(context->device, stagingBufferMemory, nullptr);

    // Step 8: Update the tensor metadata
    tensor->data = nullptr;  // Data is now on the GPU
    tensor->device = (char*)malloc(strlen("vulkan") + 1);
    strcpy(tensor->device, "vulkan");

    printf("Tensor successfully transferred to Vulkan GPU.\n");
}

void vulkan_to_cpu(Tensor* tensor) {
    VulkanContext* context = getVulkanContext();

    // Step 1: Allocate memory for CPU to hold the tensor data
    float* data_tmp = (float*)malloc(tensor->size * sizeof(float));
    if (data_tmp == NULL) {
        fprintf(stderr, "Failed to allocate memory on CPU\n");
        return;
    }

    // Step 2: Create a staging buffer (host-visible) to copy data from the Vulkan buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(context->device, context->physicalDevice, tensor->size * sizeof(float),
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    // Step 3: Copy data from the Vulkan buffer to the staging buffer
    copyBuffer(context->device, context->commandPool, context->queue, tensor->buffer, stagingBuffer, tensor->size * sizeof(float));

    // Step 4: Map the staging buffer memory to a pointer so we can access it on the CPU
    void* mappedData;
    vkMapMemory(context->device, stagingBufferMemory, 0, tensor->size * sizeof(float), 0, &mappedData);

    // Step 5: Copy data from the mapped staging buffer to the CPU memory
    memcpy(data_tmp, mappedData, tensor->size * sizeof(float));

    // Unmap the staging buffer memory
    vkUnmapMemory(context->device, stagingBufferMemory);

    // Step 6: Free the Vulkan buffer (GPU memory)
    vkDestroyBuffer(context->device, tensor->buffer, nullptr);
    vkFreeMemory(context->device, tensor->memory, nullptr);

    // Step 7: Free the staging buffer
    vkDestroyBuffer(context->device, stagingBuffer, nullptr);
    vkFreeMemory(context->device, stagingBufferMemory, nullptr);

    // Step 8: Update the Tensor structure to point to the CPU memory
    tensor->data = data_tmp;  // Now the data is on the CPU

    // Step 9: Update the device information
    const char* device_str = "cpu";
    tensor->device = (char*)malloc(strlen(device_str) + 1);
    strcpy(tensor->device, device_str);

    printf("Successfully transferred tensor to: %s\n", tensor->device);
}


void add_tensor_vulkan(Tensor* tensor1, Tensor* tensor2, Tensor* result_tensor) {
    VulkanContext* context = getVulkanContext();
    // Step 1: Ensure tensors are on Vulkan
    if (strcmp(tensor1->device, "vulkan") != 0 || strcmp(tensor2->device, "vulkan") != 0) {
        fprintf(stderr, "Tensors must be on Vulkan to perform addition\n");
        return;
    }

    // Step 2: Load the compute shader
    VkShaderModule shaderModule = loadShaderModule(context->device, "add_tensor.spv");

    // Step 3: Create descriptor sets for the buffers
    VkDescriptorSetLayoutBinding bindings[3] = {};
    
    // Binding 0: tensor1
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[0].pImmutableSamplers = nullptr;

    // Binding 1: tensor2
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].pImmutableSamplers = nullptr;

    // Binding 2: result_tensor
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].pImmutableSamplers = nullptr;

    // Step 4: Create the descriptor set layout
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descriptorSetLayout;
    vkCreateDescriptorSetLayout(context->device, &layoutInfo, nullptr, &descriptorSetLayout);

    // Step 5: Create the pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VkPipelineLayout pipelineLayout;
    vkCreatePipelineLayout(context->device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

    // Step 6: Create the compute pipeline
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";  // Entry point in shader
    pipelineInfo.layout = pipelineLayout;

    VkPipeline computePipeline;
    vkCreateComputePipelines(context->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);

    // Step 7: Create descriptor sets and allocate memory for them
    VkDescriptorSet descriptorSet;
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = context->descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    vkAllocateDescriptorSets(context->device, &allocInfo, &descriptorSet);

    // Step 8: Update descriptor sets with the buffers (tensor1, tensor2, result)
    VkDescriptorBufferInfo bufferInfos[3] = {};
    bufferInfos[0].buffer = tensor1->buffer;
    bufferInfos[0].offset = 0;
    bufferInfos[0].range = VK_WHOLE_SIZE;

    bufferInfos[1].buffer = tensor2->buffer;
    bufferInfos[1].offset = 0;
    bufferInfos[1].range = VK_WHOLE_SIZE;

    bufferInfos[2].buffer = result_tensor->buffer;
    bufferInfos[2].offset = 0;
    bufferInfos[2].range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet descriptorWrites[3] = {};
    for (int i = 0; i < 3; i++) {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }

    vkUpdateDescriptorSets(context->device, 3, descriptorWrites, 0, nullptr);

    // Step 9: Record commands to dispatch the compute shader
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(context);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    // Dispatch the compute shader with enough workgroups to cover all elements
    vkCmdDispatch(commandBuffer, (uint32_t)ceil(tensor1->size / 256.0), 1, 1);

    endSingleTimeCommands(context, commandBuffer);

    // Step 10: Cleanup (pipeline, layout, etc.)
    vkDestroyPipeline(context->device, computePipeline, nullptr);
    vkDestroyPipelineLayout(context->device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(context->device, descriptorSetLayout, nullptr);
    vkDestroyShaderModule(context->device, shaderModule, nullptr);

    printf("Summed two tensors");
}

VkShaderModule loadShaderModule(VkDevice device, const char* filePath) {
    FILE* file = fopen(filePath, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open shader file: %s\n", filePath);
        return VK_NULL_HANDLE;
    }

    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    rewind(file);

    char* buffer = (char*)malloc(fileSize);
    fread(buffer, 1, fileSize, file);
    fclose(file);

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = fileSize;
    createInfo.pCode = (uint32_t*)buffer;

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create shader module\n");
        free(buffer);
        return VK_NULL_HANDLE;
    }

    free(buffer);
    return shaderModule;
}

VkDescriptorPool createDescriptorPool(VkDevice device) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;  // Number of buffers in use

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;  // Only one descriptor set needed

    VkDescriptorPool descriptorPool;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create descriptor pool\n");
        exit(1);
    }

    return descriptorPool;
}

VkCommandBuffer beginSingleTimeCommands(VulkanContext* context) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = context->commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(context->device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void endSingleTimeCommands(VulkanContext* context, VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(context->queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(context->queue);

    vkFreeCommandBuffers(context->device, context->commandPool, 1, &commandBuffer);
}


// Clean up Vulkan resources for a tensor
void cleanup_tensor_vulkan(Tensor* tensor) {
    VulkanContext* context = getVulkanContext();
    if (tensor->device != NULL && strcmp(tensor->device, "vulkan") == 0) {
        vkDestroyBuffer(context->device, tensor->buffer, nullptr);
        vkFreeMemory(context->device, tensor->memory, nullptr);
        tensor->buffer = VK_NULL_HANDLE;
        tensor->memory = VK_NULL_HANDLE;
        tensor->device = NULL;

        printf("Tensor Vulkan resources cleaned up.\n");
    }
}

// Create a buffer and allocate memory
VkResult createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);

    return VK_SUCCESS;
}

// Copy data from one buffer to another
void copyBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

// Find a memory type that fits the requirements
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    fprintf(stderr, "Failed to find suitable memory type\n");
    exit(1);
}

VkInstance createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Tensor App";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create Vulkan instance\n");
        exit(1);
    }

    return instance;
}

VkPhysicalDevice pickPhysicalDevice(VkInstance instance) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        fprintf(stderr, "Failed to find GPUs with Vulkan support\n");
        exit(1);
    }

    VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(deviceCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);

    VkPhysicalDevice physicalDevice = devices[0];  // Pick the first device for simplicity
    free(devices);

    return physicalDevice;
}

VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, VkQueue* queue) {
    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = 0;  // Assuming 0 for simplicity
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;

    VkDevice device;
    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create logical device\n");
        exit(1);
    }

    vkGetDeviceQueue(device, 0, 0, queue);  // Get the device queue

    return device;
}

VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = 0;

    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create command pool\n");
        exit(1);
    }

    return commandPool;
}


