import torch
from torch import Tensor

# #####################################################
# =*= TENSOR-RELATED TOOLS USED ACCROSS THE PROJECT =*=
# #####################################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

def add_into_tensor(tensor_list: Tensor | None, tensor_val: Tensor):
    if tensor_list is None:
        tensor_list = tensor_val
    else:
        tensor_list = torch.cat((tensor_list, tensor_val), dim=0)
    return tensor_list

def to_binary(booleanVal: bool):
    return 1 if booleanVal else 0

def features2tensor(features: list, device: str):
    for f in features:
        if isinstance(f, bool):
            f = to_binary(f)
    return torch.tensor([[f for f in features]], dtype=torch.float32, device=device)

def id2tensor(id1: int, id2: int, device: str):
    return torch.tensor([[id1], [id2]], dtype=torch.long, device=device)

def move_tensors(obj, device, path='root'):
    """
        Recursively move all torch.Tensor objects in `obj` to the specified device.
    """
    if isinstance(obj, Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_tensors(item, device, f'{path}[{i}]') for i, item in enumerate(obj)]
    elif isinstance(obj, tuple):
        return tuple(move_tensors(item, device, f'{path}[{i}]') for i, item in enumerate(obj))
    elif isinstance(obj, set):
        return {move_tensors(item, device, f'{path}[{i}]') for i, item in enumerate(obj)}
    elif isinstance(obj, dict):
        return {k: move_tensors(v, device, f"{path}['{k}']") for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        for attr_name, attr_value in vars(obj).items():
            moved_value = move_tensors(attr_value, device, f'{path}.{attr_name}')
            setattr(obj, attr_name, moved_value)
        return obj
    else:
        return obj

def scatter_sum(src, index, dim, dim_size):
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    out.index_add_(dim, index, src)
    return out

def fast_self_attention(attention_coefs: Tensor, non_linear_activation, matrix: Tensor):
    coef1, coef2 = attention_coefs.chunk(2, dim=0)
    w_sum        = coef1 + coef2
    return non_linear_activation(matrix @ w_sum)

def fast_cross_attention(attention_coefs: Tensor, non_linear_activation, m1: Tensor, m2: Tensor):
    coef1, coef2 = attention_coefs.split(m1.size(-1), dim=0)
    return non_linear_activation((m1 @ coef1) + (m2 @ coef2))
