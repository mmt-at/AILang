from models.attention import AilangSelfAttention
from models.attention import TorchSelfAttention
import ailang as al
import numpy as np
import torch
from ailang.util import check_intermediate, create_numpy_params


def test_attention():
    batch_size = 1
    seq_len = 6
    d_model = 3
    num_heads = 1
    t_model = TorchSelfAttention(d_model=3, num_heads=1)
    pd = create_numpy_params(t_model)
    a_model = AilangSelfAttention(pd, d_model, num_heads).mlu()
    x = np.random.randn(seq_len, d_model).astype(np.float32)
    a = al.from_numpy(x)
    t = torch.from_numpy(x)
    a_model.eval()
    t_model.eval()
    a_output = a_model(a)
    t_output = t_model(t)
    if not check_intermediate(a_output, t_output, 'final_output'):
        print("Final output does not match.")
        return
    
    print("Outputs match!")

if __name__ == "__main__":
    test_attention()
