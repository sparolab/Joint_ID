
from mmcv.utils import Registry
import torch.optim

OPTIMIZER = Registry('optimizer')

@OPTIMIZER.register_module()
class Adam(torch.optim.Adam):
    def __init__(self, **kwargs):
        super(Adam, self).__init__(**kwargs)
        
        
@OPTIMIZER.register_module()
class AdamW(torch.optim.AdamW):
    def __init__(self, **kwargs):
        super(AdamW, self).__init__(**kwargs)


@OPTIMIZER.register_module()
class SGD(torch.optim.SGD):
    def __init__(self, **kwargs):
        super(SGD, self).__init__(**kwargs)



