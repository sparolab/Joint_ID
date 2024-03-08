
import torch
import torch.nn.functional as F
import torchsummaryX
import torch.nn as nn
import warnings



def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def _transform_inputs(inputs, in_index, input_transform, align_corners, scale_factor=1):

    if input_transform == 'resize_concat':
        input_list = [inputs[i] for i in in_index]
        upsampled_inputs = [
            resize(
                input=x,
                size=input_list[0].shape[2:],
                mode='bilinear',
                scale_factor=scale_factor,
                align_corners=align_corners) for x in input_list
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)
        
    elif input_transform == 'multiple_select':
        input_list = [inputs[i] for i in in_index]
        
    else:
        input_list = inputs[in_index]

    return input_list