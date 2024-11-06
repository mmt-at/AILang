import torch
import ailang as al
import ailang.nn as nn
import numpy as np
import torch.nn.functional as F

class TorchSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model必须能够被num_heads整除"

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.fc_out = torch.nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        seq_len, d_model = x.shape  # 假设 x 的形状为 (seq_len, d_model)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        sqrt_d_k = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        qk = torch.matmul(Q, K.transpose(-2, -1)) / sqrt_d_k
        attention_weights = F.softmax(qk, dim=-1)
        out = torch.matmul(attention_weights, V)
        return out


class AilangSelfAttention(nn.Module):
    def __init__(self, pd, d_model, num_heads=1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model必须能够被num_heads整除"

        self.query = nn.Linear(d_model, d_model).npu()
        self.query.weight = torch.nn.Parameter(al.from_numpy(pd["query.weight"]))
        self.query.bias = torch.nn.Parameter(al.from_numpy(pd["query.bias"]))
        self.key = nn.Linear(d_model, d_model).npu()
        self.key.weight = torch.nn.Parameter(al.from_numpy(pd["key.weight"]))
        self.key.bias = torch.nn.Parameter(al.from_numpy(pd["key.bias"]))
        self.value = nn.Linear(d_model, d_model).npu()
        self.value.weight = torch.nn.Parameter(al.from_numpy(pd["value.weight"]))
        self.value.bias = torch.nn.Parameter(al.from_numpy(pd["value.bias"]))
        self.fc_out = nn.Linear(d_model, d_model).npu()
        self.fc_out.weight = torch.nn.Parameter(al.from_numpy(pd["fc_out.weight"]))
        self.fc_out.bias = torch.nn.Parameter(al.from_numpy(pd["fc_out.bias"]))

    def forward(self, x, mask=None):
        seq_len, d_model = x.shape  # 6 3
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        r = al.from_numpy(np.full((seq_len, seq_len), self.d_k).astype(np.float32))
        sqrt = al.sqrt(r)
        kt = al.transpose(K, 1, 0)
        qk = al.matmul(Q, kt)
        scores = al.div(qk, sqrt)
        attention_weights = al.softmax(scores, dim=-1)
        out = al.matmul(attention_weights, V)
        return out

