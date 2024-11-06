import torch.nn as nn
import torch
# from torch import Tensor
import os
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class TorchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.LSTMCell(input_size, hidden_size))
        for i in range(num_layers - 1):
            self.layers.append(nn.LSTMCell(hidden_size, hidden_size))
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, inputs):  # seq_len, batch, input_size
        batch_size = inputs.shape[1]
        state_c = [torch.zeros(batch_size, self.hidden_size, device="cpu") for _ in range(10)] # hardcode for ts compile
        state_h = [torch.zeros(batch_size, self.hidden_size, device="cpu") for _ in range(10)]
        for i in range(inputs.size()[0]):
            cur_input = inputs[i]
            for j, layer in enumerate(self.layers):
                c = state_c[j]
                h = state_h[j]
                c, h = layer(cur_input, (c, h))

                state_c[j] = c
                state_h[j] = h
                cur_input = h
        return state_h[self.num_layers - 1]


import ailang as al
from ailang import nn
import ailang.nn as nn

class AilangLSTM(al.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pd):
        super().__init__()
        self.layers = al.nn.ModuleList()
        self.layers.append(al.nn.LSTMCell(input_size, hidden_size))
        for i in range(num_layers - 1):
            self.layers.append(al.nn.LSTMCell(hidden_size, hidden_size))
        # 从pd中加载上面TorchLSTM中的参数
        for i, layer in enumerate(self.layers):
            layer.weight_ih = al.from_numpy(pd[f"layers.{i}.weight_ih"])
            layer.weight_hh = al.from_numpy(pd[f"layers.{i}.weight_hh"])
            layer.bias_ih = al.from_numpy(pd[f"layers.{i}.bias_ih"])
            layer.bias_hh = al.from_numpy(pd[f"layers.{i}.bias_hh"])
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, inputs):  # seq_len, batch, input_size
        batch_size = inputs.shape[1]
        state_c = [al.zeros(batch_size, self.hidden_size, device="npu") for _ in range(10)] # hardcode for ts compile
        state_h = [al.zeros(batch_size, self.hidden_size, device="npu") for _ in range(10)]
        for i in range(inputs.size()[0]):
            cur_input = inputs[i]
            for j, layer in enumerate(self.layers):
                c = state_c[j]
                h = state_h[j]
                c, h = layer(cur_input, (c, h))

                state_c[j] = c
                state_h[j] = h
                cur_input = h
        return state_h[self.num_layers - 1]