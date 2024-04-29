from typing import Dict
import torch.nn as nn
import torch
from torch.ao.quantization import QConfig
from .utils import (quantize_from_qparams,
                    dtype_to_bw, linear_on_quantized_data, 
                    create_qparams_from_dtype)

all = [
    'QuantizedLinear'
]

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, w_qparams=None, a_qparams=None) -> None:
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.w_qparams = w_qparams
        self.a_qparams = a_qparams
        if self.w_qparams is not None:
            self.register_buffer("weight_scales", self.w_qparams.scales)
            self.register_buffer("weight_zero_points", 
                                 self.w_qparams.zero_points)
        if self.a_qparams is not None:
            self.register_buffer("act_scales", self.a_qparams.scales)
            self.register_buffer("act_zero_points", self.a_qparams.zero_points)
    
    @classmethod
    def from_float(cls, float_mod):
        assert hasattr(float_mod, 'qconfig') and isinstance(float_mod.qconfig, 
                                                            QConfig)
        weight_process = float_mod.qconfig.weight()
        w_dtype = weight_process.dtype
        num_kernels = float_mod.weight.shape[0]
        device=float_mod.weight.device

        w_qparams = create_qparams_from_dtype(dtype=w_dtype, 
                                                device=device,
                                                is_channel_wise=True,
                                                num_kernels=num_kernels
                                                )
                                              

        act_process = float_mod.qconfig.activation()
        act_dtype = act_process.dtype

        a_qparams = create_qparams_from_dtype(dtype=act_dtype,
                                              device=device)


        new_mod = cls(float_mod.in_features,
                      float_mod.out_features,
                      float_mod.bias is not None,
                      device=float_mod.weight.device,
                      w_qparams=w_qparams,
                      a_qparams=a_qparams
                      )

        weight = float_mod.weight.detach()
        if w_qparams is not None:
            weight = quantize_from_qparams(weight, w_qparams)
        new_mod.register_buffer("weight", weight)
        if float_mod.bias is not None:
            bias = float_mod.bias.detach()
            new_mod.register_buffer("bias", bias)
        else:
            new_mod.bias = None
        return new_mod
    
    def _get_name(self):
        w_width = 16 if self.w_qparams is None else \
                  dtype_to_bw[self.w_qparams.dtype]
        a_width = 16 if self.a_qparams is None else \
                  dtype_to_bw[self.a_qparams.dtype]
        return f"QuantizedLinear(W({w_width})A({a_width}))"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.a_qparams is not None and x.dtype == torch.float16:
            x = quantize_from_qparams(x, self.a_qparams)
        return linear_on_quantized_data(self.weight, self.w_qparams, x, 
                                        self.a_qparams, self.bias)
