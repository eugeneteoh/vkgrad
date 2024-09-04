import ctypes
import os


class CTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int),
    ]

class Tensor:
    os.path.abspath(os.curdir)
    _C = ctypes.CDLL("libtensor.so")

    def __init__(self, data=None, device="cpu"):
        if data is None:
            self.tensor = None,
            self.shape = None,
            self.ndim = None,
            self.device = None
        else:
            if isinstance(data, (float, int)):
                data = [data]

            data, shape = self.flatten(data)
            self.data_ctype = (ctypes.c_float * len(data))(*data)
            self.shape_ctype = (ctypes.c_int * len(shape))(*shape)
            self.ndim_ctype = ctypes.c_int(len(shape))
            self.device_ctype = device.encode("utf-8")

            self.shape = shape
            self.ndim = len(shape)
            self.device = device

            Tensor._C.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p]
            Tensor._C.create_tensor.restype = ctypes.POINTER(CTensor)

            self.tensor = Tensor._C.create_tensor(self.data_ctype, self.shape_ctype, self.ndim_ctype, self.device_ctype)

    def flatten(self, nested_list):
        def recursive_flatten(nested_list):
            flat_data = []
            shape = []
            if isinstance(nested_list, list):
                for sublist in nested_list:
                    inner_data, inner_shape = recursive_flatten(sublist)
                    flat_data.extend(inner_data)
                shape.append(len(nested_list))
                shape.extend(inner_shape)
            else:
                flat_data.append(nested_list)
            return flat_data, shape

        flat_data, shape = recursive_flatten(nested_list)
        return flat_data, shape

    def __getitem__(self, indices):
        if len(indices) != self.ndim:
            raise ValueError("Number of indices must match the number of dimensions")

        Tensor._C.get_item.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int)]
        Tensor._C.get_item.restype = ctypes.c_float

        indices = (ctypes.c_int * len(indices))(*indices)
        value = Tensor._C.get_item(self.tensor, indices)
        return value


    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("Tensors must have the same shape for addition")

        Tensor._C.add_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
        Tensor._C.add_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.add_tensor(self.tensor, other.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim
        result_data.device = self.device

        return result_data

    def to(self, device):
        self.device = device
        self.device_ctype = self.device.encode("utf-8")

        Tensor._C.to_device.argtypes = [ctypes.POINTER(CTensor), ctypes.c_char_p]
        Tensor._C.to_device.restype = None
        Tensor._C.to_device(self.tensor, self.device_ctype)
    
        return self

        
tensor1 = Tensor([[1, 2, 3], [3, 2, 1]])
tensor2 = Tensor([[3, 2, 1], [1, 2, 3]])
tensor3 = tensor1 + tensor2

print(tensor1.shape)
print(tensor1[0, 0])
print(tensor3[0, 0])
print(tensor1.device)

tensor1.to("vulkan")
tensor1.to("cpu")

tensor1.to("vulkan")
tensor2.to("vulkan")
tensor3 = tensor1 + tensor2
tensor3.to("cpu")
print(tensor3.tensor)
# print(tensor3[0, 0])


def print_data(self):
    if self.tensor is None:
        print("Tensor is empty.")
        return

    # Iterate over the data using the size attribute to know how many elements to print
    print("Tensor data:")
    for i in range(self.tensor.contents.size):
        print(self.tensor.contents.data[i])



print_data(tensor3)