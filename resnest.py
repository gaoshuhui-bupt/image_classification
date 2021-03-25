##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""

import torch
from resnet import ResNet, Bottleneck

# __all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']
__all__ = ['resnest50']

_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50')
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def resnest50(pretrained=True, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
        radix=2, groups=1, bottleneck_width=64,
        deep_stem=True, stem_width=32, avg_down=True,
        avd=True, avd_first=False, **kwargs)
    if pretrained:
        # print(torch.hub.load_state_dict_from_url(
        #     resnest_model_urls['resnest50'], progress=True, check_hash=True))
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True))
    return model


# resnest = resnest50()
# print(resnest)

