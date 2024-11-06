import torch
import torch_npu

def from_numpy(array):
    # 将 NumPy 数组转换为 torch.Tensor
    tensor = torch.from_numpy(array).npu()
    # 将 tensor 转换为 torch.nn.Parameter
    return torch.nn.Parameter(tensor)