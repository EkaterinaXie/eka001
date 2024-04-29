import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

import argparse
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline
from torch.ao.quantization import QConfig, MinMaxObserver, PlaceholderObserver, QuantStub, DeQuantStub
from quant_ops.Linear import QuantizedLinear
from quant_ops.Conv2d import QuantizedConv2d
from quant_ops.quantizer_dequantizer import Quantizer, DeQuantizer
from quant_ops.utils import QParam


# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009) 

from torch.ao.quantization import QuantStub, DeQuantStub

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t)) # feature列表
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self, is_qat=False):
        fuse_modules = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
        for m in self.modules():
            if type(m) == ConvBNReLU: # if type of child_module is ConvBNReLU
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:# if type of child_module is InvertedResidual
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def prepare_data_loaders(data_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
        data_path, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageNet(
        data_path, split="val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)
    
    return data_loader, data_loader_test

def print_each_layer_bitwise(model):
    torch.save(model.state_dict(), "temp.p")
    # print(model.state_dict())
    state_dict = model.state_dict()

    for key, value in state_dict.items():
        if key in ["classifier.1._packed_params.dtype", "classifier.1._packed_params._packed_params"]:
            print(f"Key:{key}")
        else:
            print(f"Key: {key}")
            print(f"Value: {value.dtype}")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def convert_to_quantized(model):
    from torch.ao.quantization import convert
    convert(model, 
            mapping={torch.nn.modules.linear.Linear: QuantizedLinear,
                     torch.nn.modules.conv.Conv2d: QuantizedConv2d,
                     torch.ao.quantization.stubs.QuantStub: Quantizer,
                     torch.ao.quantization.stubs.DeQuantStub: DeQuantizer
                    },
            inplace=True)

'''Customize the bit width of each layer of the network '''
def custom_each_layer_bitwise(Model, a_conv_wid, w_conv_wid, a_ln_wid, w_ln_wid):
    # custom_layers = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Identity]
    # custom_layers = {
    #     'conv':torch.nn.modules.conv.Conv2d,
    #     'linear':torch.nn.modules.linear.Identity,
    # }

    conv_activate_bit = a_conv_wid
    conv_weight_bit = w_conv_wid
    linear_activate_bit = a_ln_wid
    linear_weight_bit = w_ln_wid

    bw_to_dtype = {
        8: torch.quint8,
        4: torch.quint4x2,
        2: torch.quint4x2, # !!!TODO: 2 is not supported, treat as 4
    }
    
    for name, module in Model.named_modules():
        # conv layer
        if type(module) == torch.nn.modules.conv.Conv2d:
            a_dtype_conv = bw_to_dtype[conv_activate_bit]
            w_dtype_conv = bw_to_dtype[conv_weight_bit]
            act_conv = PlaceholderObserver.with_args(dtype=a_dtype_conv)
            weight_conv = PlaceholderObserver.with_args(dtype=w_dtype_conv)
            module.qconfig = torch.ao.quantization.QConfig(activation=act_conv, weight=weight_conv)
        elif type(module) == torch.nn.modules.linear.Linear:
            a_dtype_linear = bw_to_dtype[linear_activate_bit]
            w_dtype_linear = bw_to_dtype[linear_weight_bit]
            act_linear = PlaceholderObserver.with_args(dtype=a_dtype_linear)
            weight_linear = PlaceholderObserver.with_args(dtype=w_dtype_linear)
            module.qconfig = torch.ao.quantization.QConfig(activation=act_linear, weight=weight_linear)
        else:
            module.qconfig = torch.ao.quantization.default_qconfig
            # module.qconfig = torch.ao.quantization.QConfig(activation=torch.float16, weight=torch.float16)

if __name__ == "__main__":
    # Load the pre-trained MobileNetV2 model
    data_path = '/share/xierui-nfs/dataset/imageNet-1k'
    saved_model_dir = '/share/xierui-nfs/pythonProgram/mobilenet/'
    # load pre-trained parameters
    float_model_file = 'mobilenet_v2-b0353104.pth'

    train_batch_size = 30
    eval_batch_size = 50

    data_loader, data_loader_test = prepare_data_loaders(data_path)
    criterion = nn.CrossEntropyLoss()

    num_eval_batches = 1
    num_calibration_batches = 32 

    '''print each layer's name and type'''
    # for name, module in per_channel_quantized_model.named_modules():
    #     print(f"Layer name: {name}, Layer type: {type(module)}")

    per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
    per_channel_quantized_model.eval()
    # per_channel_quantized_model.fuse_model() # fuse Conv/BN/ReLU layer or not
    custom_each_layer_bitwise(per_channel_quantized_model, 4, 4, 8, 8)

    torch.ao.quantization.prepare(per_channel_quantized_model, inplace=True)

    '''Call our custom bit-width function'''
    convert_to_quantized(per_channel_quantized_model)

    print("Size of model after quantization")
    print_each_layer_bitwise(per_channel_quantized_model)