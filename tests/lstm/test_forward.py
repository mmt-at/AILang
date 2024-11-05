from models.lstm import AilangLSTM
from models.lstm import TorchLSTM
import ailang as al
import numpy as np
import torch
from ailang.util import check_intermediate, create_numpy_params

def test_resnet():
    
    # print(pd.keys())
    t_model = TorchLSTM(256,256,10)
    pd = create_numpy_params(t_model)
    
    a_model = AilangLSTM(256,256,10,pd).mlu()
    x = np.random.randn(64, 1, 256).astype(np.float32)
    t = torch.from_numpy(x)
    a = al.from_numpy(x)
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
