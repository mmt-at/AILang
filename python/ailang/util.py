import logging
import numpy as np
import torch
# 配置日志记录
logging.basicConfig(
    filename='debug_logs.txt',  # 日志文件名
    filemode='w',               # 写入模式，'w'表示覆盖写入，'a'表示追加
    level=logging.INFO,         # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)

def create_numpy_params(model):
    """为模型生成一个 NumPy 随机参数字典"""
    params_dict = {}
    model_params = (
        model.named_parameters()
        if isinstance(model, torch.nn.Module)
        else model.parameters()
    )
    for name, param in model_params:
        params_dict[name] = param.detach().numpy()
    return params_dict


def check_intermediate(a_out, t_out, layer_name, rtol=1e-06, atol=1e-09):
    a_out_np = a_out.detach().cpu().numpy()
    t_out_np = t_out.detach().cpu().numpy()
    # 计算误差
    diff = np.abs(a_out_np - t_out_np)
    # 找到误差超过阈值的位置
    mismatch_indices = np.where(diff > (atol + rtol * np.abs(t_out_np)))
    if mismatch_indices[0].size > 0:
        logging.info(f"Mismatch at {layer_name}")
        
        # 获取所有不匹配位置的误差值
        all_errors = [(idx, diff[idx]) for idx in zip(*mismatch_indices)]
        # 按误差大小降序排序
        all_errors.sort(key=lambda x: x[1], reverse=True)
        
        # 首先输出最大误差
        max_error_idx, max_error = all_errors[0]
        logging.info(f"最大误差 - Index {max_error_idx}: AILang output={a_out_np[max_error_idx]}, "
                    f"PyTorch output={t_out_np[max_error_idx]}, Error={max_error}")
        
        # 然后输出其他误差
        for idx, error in all_errors[1:]:
            log_message = f"Index {idx}: AILang output={a_out_np[idx]}, PyTorch output={t_out_np[idx]}, Error={error}"
            logging.info(log_message)
            if len(all_errors) > 100 and all_errors.index((idx, error)) >= 99:
                logging.info("显示前100个不匹配项。")
                break
        return False
    return True
