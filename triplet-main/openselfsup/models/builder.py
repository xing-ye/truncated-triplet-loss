from torch import nn

from openselfsup.utils import build_from_cfg
from .registry import (BACKBONES, MODELS, NECKS, HEADS, MEMORIES, LOSSES)


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): 模块的配置，它可以是字典或配置列表。
        registry (:obj:`Registry`):模块所属的注册表
        default_args (dict, optional): Default arguments to build the module.
            Default: None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_memory(cfg):
    """Build memory."""
    return build(cfg, MEMORIES)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_model(cfg):
    """Build model."""
    return build(cfg, MODELS)
