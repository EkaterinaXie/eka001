from collections import namedtuple
import torch
from typing import Optional
import torch.nn.functional as F
import math

SCALE=0.1  # init the quant parameters


def quantize_per_tensor_uint4(
    input: torch.Tensor, scale, zero_point, 
):
    scale_inv = 1.0 / scale
    int_repr = torch.clamp(torch.round(input * scale_inv) + zero_point, 0, 15).to(torch.uint8)
    if len(input.shape) >= 4:
        # if input.shape[1] % 2 != 0:
        #     print(f"input.shape_long_input_!=0 = {input.shape}")
        # elif input.shape[1] % 2 == 0:
        #     print(f"input.shape_long_input_==0 = {input.shape}")
        # assert input.shape[1] % 2 == 0
        return (int_repr[:, ::2, ...] << 4 | int_repr[:, 1::2, ...])
    # print(f"input.shape_short_input = {input.shape}")
    # assert input.shape[-1] % 2 == 0
    return (int_repr[...,::2] << 4 | int_repr[..., 1::2])


def unpack_uint4(input):
    shape = input.shape
    if len(shape) >= 4:
        packed_dim = 2
        new_shape = (input.shape[0], input.shape[1]*2, *input.shape[2:])
    else:
        packed_dim = -1
        new_shape = (*input.shape[:-1], input.shape[-1]*2)
    first_elements = (input >> 4).to(torch.uint8)
    second_elements = (input & 0b1111).to(torch.uint8)
    return torch.stack([first_elements, second_elements], dim=packed_dim).view(new_shape)
    
    
def dequantize_per_tensor_uint4(
        input, scale, zero_point,
):
    input = unpack_uint4(input)
    return (input.view(torch.uint8).to(torch.float32) - zero_point) * scale


dtype_to_bw = {
    torch.quint8:8,
    torch.quint4x2:4,
    torch.quint2x4:2,
    torch.float16:16,
}

class QParam(namedtuple("QParam", 
                        ["qscheme", "dtype", "scales", "zero_points", "axis"],
                        defaults=[torch.per_tensor_affine, torch.qint8,
                                  1.0, 0.0, 0])):
    pass

def create_qparams_from_dtype(  
                            dtype, 
                            device, 
                            is_channel_wise=False, 
                            num_kernels=None,
):

    if dtype == torch.float16:
        return None
    elif dtype in [torch.qint8, torch.quint8]:
        if is_channel_wise:
            assert num_kernels is not None
            scales = SCALE * torch.ones((num_kernels,),
                                        dtype=torch.float16,
                                        device=device)
            zero_points = torch.zeros_like(scales)
            return QParam(qscheme=torch.per_channel_affine,
                        scales=scales, zero_points=zero_points,
                        dtype=dtype, axis=0)
        else:
            scales = torch.tensor(SCALE, dtype=torch.float16, device=device)
            zero_points = torch.zeros_like(scales)
            return QParam(qscheme=torch.per_tensor_affine,
                        scales=scales, zero_points=zero_points,
                        dtype=dtype)
    elif dtype in [torch.quint4x2]:
        scales = torch.tensor(SCALE, dtype=torch.float16, device=device)
        zero_points = torch.zeros_like(scales)
        return QParam(qscheme=torch.per_tensor_affine,
                        scales=scales, zero_points=zero_points,
                        dtype=dtype)
    else:
        raise ValueError(f"Unsupported quantize dtype {dtype}")


def quantize_from_qparams(x: torch.Tensor, qparams: QParam):
    if qparams.dtype == torch.quint4x2:
        assert qparams.qscheme == torch.per_tensor_affine
        return quantize_per_tensor_uint4(x, SCALE, 0)
    
    if qparams.qscheme in [torch.per_tensor_affine]:
        scales = qparams.scales
        scales = scales.clone().detach().to(x.device) \
                 if isinstance(scales, torch.Tensor) \
                 else torch.tensor(scales, dtype=torch.float16, device=x.device)
        zps = qparams.zero_points
        zps = zps.clone().detach().to(x.device) \
              if isinstance(zps, torch.Tensor) \
              else torch.tensor(zps, dtype=torch.float16, device=x.device)
        # Quantize only works on Float Tensor not Half. TODO: custom kernels
        x = x.to(torch.float32)
        x_quant = torch.quantize_per_tensor(x, scales, zps, qparams.dtype)
    elif qparams.qscheme in [torch.per_channel_affine]:
        scales = qparams.scales
        assert isinstance(scales, torch.Tensor)
        scales = scales.clone().detach().to(x.device)
        zps = qparams.zero_points
        assert isinstance(zps, torch.Tensor)
        zps = zps.clone().detach().to(x.device)
        assert qparams.axis < len(x.shape)
        # Quantize only works on Float Tensor not Half TODO: custom kernels
        x = x.to(torch.float32)
        x_quant = torch.quantize_per_channel(x, scales, zps, axis=qparams.axis,
                                             dtype=qparams.dtype)
    else:
        raise ValueError(f"Unknown qscheme {qparams.qscheme}")
    return x_quant


def dequantize_to_float16(x: torch.Tensor):
    if x.dtype == torch.float16:
        return x
    if x.dtype in [torch.quint8, torch.qint8]:
        return x.dequantize().to(torch.float16)
    assert x.dtype == torch.uint8 # the current way to support uint4
    # return dequantize_per_tensor_uint4(x, qparams.scales.to(x.device), qparams.zero_points.to(x.device)).to(torch.float16)
    return dequantize_per_tensor_uint4(x, SCALE, 0).to(torch.float16)

def linear_on_quantized_data(
        w_tensor: torch.Tensor,
        w_qparams: QParam,
        a_tensor: torch.Tensor,
        a_qparams: QParam,
        bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # functional simulation for now (TODO: kernel support)
    a_tensor = dequantize_to_float16(a_tensor)
    w_tensor = dequantize_to_float16(w_tensor)
    return F.linear(a_tensor, w_tensor, bias)


def conv2d_on_quantized_data(
        w_tensor: torch.Tensor = None,
        w_qparams: QParam = None,
        a_tensor: torch.Tensor = None,
        a_qparams: QParam = None,
        bias: Optional[torch.Tensor] = None,
        stride=1,
        padding=0, 
        dilation=1,
        groups=1
) -> torch.Tensor:
    # functional simulation for now (TODO: kernel support)
    a_tensor = dequantize_to_float16(a_tensor)
    w_tensor = dequantize_to_float16(w_tensor)
    return F.conv2d(a_tensor, w_tensor, bias, stride, padding, dilation, groups)
