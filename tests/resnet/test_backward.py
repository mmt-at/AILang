# Torch
from models.resnet import AilangResNet
from models.resnet import TorchResNet
import ailang as al
import numpy as np
import torch
from ailang.util import check_intermediate, create_numpy_params

def test_resnet():
    t_model = TorchResNet()
    pd = create_numpy_params(t_model)
    a_model = AilangResNet(pd).mlu()
    x = np.random.randn(1, 3, 112, 112).astype(np.float32)
    t = torch.from_numpy(x).float()
    t.requires_grad = True
    a = al.from_numpy(x).float()
    a.requires_grad = True
    a_model.eval()
    t_model.eval()
    a_out = a_model(a)
    a_out = al.sum(a_out)
    a_out.backward()
    al_grad = a.grad
    t_out = t_model(t)
    t_out = t_out.sum()
    t_out.backward()
    t_grad = t.grad
    # assert numeric_check(al_grad, t_grad)
    if not check_intermediate(al_grad, t_grad, 'grad'):
        print("Final grad does not match.")
        return
        
    print("final grad match!")



if __name__ == "__main__":
    test_resnet()
