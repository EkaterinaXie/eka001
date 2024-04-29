import torch.nn as nn
import torch
from torch.ao.quantization import QConfig
from .utils import (quantize_from_qparams, create_qparams_from_dtype,
                    conv2d_on_quantized_data, dtype_to_bw)


all = [
    'QuantizedConv2d'
]


class QuantizedConv2d(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 device=None,
                 w_qparams=None, a_qparams=None) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
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
        # init the w & a quant parameters
        w_qparams = create_qparams_from_dtype(dtype=w_dtype, 
                                                device=device,
                                                is_channel_wise=True,
                                                num_kernels=num_kernels
                                                )


        act_process = float_mod.qconfig.activation()
        act_dtype = act_process.dtype
        
        a_qparams = create_qparams_from_dtype(dtype=act_dtype,
                                              device=device)
            
        new_mod = cls(float_mod.in_channels,
                      float_mod.out_channels,
                      float_mod.kernel_size,
                      float_mod.stride,
                      float_mod.padding,
                      float_mod.dilation,
                      float_mod.groups,
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
        return f"QuantizedConv2d(W({w_width})A({a_width}))"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.a_qparams is not None and x.dtype == torch.float16:
            x = quantize_from_qparams(x, self.a_qparams)

        return conv2d_on_quantized_data(self.weight,
                                        self.w_qparams,
                                        x,
                                        self.a_qparams,
                                        self.bias,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        self.groups
                                        )
