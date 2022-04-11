from openselfsup.utils import Registry

MODELS = Registry('model')
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
MEMORIES = Registry('memory')
LOSSES = Registry('loss')
'''
https://zhuanlan.zhihu.com/p/62481975
我猜是这个里的一个映射？
'''