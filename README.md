# custom each layer's widsize of a network
---
- network we use as an example: MobileNetv2
- customize layer type: conv, linear
- widsize we set: W4A4, W8A8

# how to set parameters?
- python main.py
- find this code: custom_each_layer_bitwise(per_channel_quantized_model, 4, 4, 8, 8)
- 4 numbers mean the widsize set for: activate_conv_wid, weight_conv_wid, activate_linear_wid, weight_linear_wid

# result
- # original
- size of model without any process: 14.244164 MB
  - W32A32, torch.float32
- size of model after fusion: 13.990612 MB
  - W32A32, torch.float32
- size of model with fusion and default widsize -- W8A8: 3.620814 MB
  - W8A8, torch.qint8
- # our
- size of model without fusion but use custom widsize -- W4A4: 2.126688 MB
  - W4A4, torch.uint8
- size of model without fusion but use custom widsize -- W8A8: 4.12763 MB
  - W8A8, torch.quint8
- # we can set widsize of activation and weight of each conv layer and linear layer.
- # examples above set all conv layers and linear layers with the same widsize.
