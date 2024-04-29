# custom each layer's widsize of a network
---
- network we use as an example: MobileNetv2
- customize layer type: conv, linear
- widsize we set: W4A4, W8A8

# how to set parameters?
- python main.py
- find this code: custom_each_layer_bitwise(per_channel_quantized_model, 4, 4, 8, 8)
- 4 numbers mean the widsize set for: conv_wid, w_conv_wid, a_ln_wid, w_ln_wid
