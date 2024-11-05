# Torch
from models.resnet import AilangResNet
from models.resnet import TorchResNet
import ailang as al
import numpy as np
import torch
from ailang.util import check_intermediate, create_numpy_params

def test_resnet():
    
    # print(pd.keys())
    t_model = TorchResNet()
    pd = create_numpy_params(t_model)
    
    a_model = AilangResNet(pd).mlu()
    x = np.random.randn(1, 3, 112, 112).astype(np.float32)
    t = torch.from_numpy(x)
    print(t.device)
    a = al.from_numpy(x)
    print(a.device)
    a_model.eval()
    t_model.eval()
    
    # 最终输出检查
    a_out = a_model(a)
    t_out = t_model(t)
    if not check_intermediate(a_out, t_out, 'final_output'):
        print("Final output does not match.")
        return
    
    print("Outputs match!")

if __name__ == "__main__":
    test_resnet()
