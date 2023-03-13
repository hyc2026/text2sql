# MIT License
#
# Copyright (c) 2019 seq2struct contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import collections
import collections.abc as abc
import inspect
import sys


# 全局注册字典
_REGISTRY = collections.defaultdict(dict)


def register(kind, name):
    """
    常用作装饰器，进行注册。注册的对象 obj 应是可调用的，即实现了 __call__ 方法
    :param kind: 类型字符串，如 model、dataset、preproc、tokenizer、optimizer、lr_scheduler 等
    :param name: 名称字符串，如 DuoRAT、Transformer、Bert 等
    :return:
    """
    # _REGISTRY[kind] 为类型注册字典 kind_registry
    kind_registry = _REGISTRY[kind]

    def decorator(obj):
        # 在类型注册字典下，避免重名
        if name in kind_registry:
            raise LookupError("{} already registered as kind {}".format(name, kind))
        # 若不重名，则进行注册，name 为键、输入的对象 obj 为值
        kind_registry[name] = obj
        return obj

    return decorator


def construct(kind, config, unused_keys=(), **kwargs):
    """
    根据输入参数构造 全局注册字典 中的某个已注册的可调用对象
    :param kind:
    :param config:
    :param unused_keys:
    :param kwargs:
    :return:
    """
    return instantiate(lookup(kind, config), config, unused_keys + ("name",), **kwargs)


def lookup(kind, name):
    """
    访问全局注册字典 _REGISTRY
    :param kind: 类型字符串
    :param name: 名称字符串
    :return:
    """
    if isinstance(name, abc.Mapping):
        name = name["name"]

    # 确保访问的 kind 已被注册
    if kind not in _REGISTRY:
        raise KeyError('Nothing registered under "{}"'.format(kind))
    # _REGISTRY[kind] 为类型注册字典，_REGISTRY[kind][name] 为所求目标对象
    # (该对象应是可调用的，即实现了 __call__ 方法)
    return _REGISTRY[kind][name]


def instantiate(_callable, config, unused_keys=(), **kwargs):
    """
    进行参数检查，并将参数输入给可调用对象
    :param _callable: 可调用对象
    :param config: 配置参数
    :param unused_keys: 指定的、不被 _callable 使用的参数(的键)
    :param kwargs: 其它键值对参数
    :return:
    """
    # 把配置参数 config 和输入给 instantiate 函数的键值对参数 **kwargs 合并给 merged
    merged = {**config, **kwargs}

    # 获得参数签名对象，signature.parameters 即为可调用对象 _callable 的全部参数字典
    # 这是一个有序字典 OrderedDict，键为参数名，值为参数值
    signature = inspect.signature(_callable)

    # 检查可调用对象 _callable 所有参数值的类型
    # POSITIONAL_ONLY 和 VAR_POSITIONAL 不支持，若出现 则报错
    for name, param in signature.parameters.items():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            raise ValueError(
                "Unsupported kind for param {}: {}".format(name, param.kind)
            )

    # 如果可调用对象 _callable 存在参数值的类型为 VAR_KEYWORD，则将全部参数 merged 传给 _callable 并返回
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return _callable(**merged)

    # 对于全部参数 merged 而言，如果其中的键 key 既不在可调用对象 _callable 的参数列表里，
    # 也不在指定的不使用的键 unused_keys 里，则表示它是一个缺失(输入了但未被使用)的参数，加入 missing 字典
    missing = {}
    for key in list(merged.keys()):
        if key not in signature.parameters:
            if key not in unused_keys:
                missing[key] = merged[key]
            merged.pop(key)
    # 如果存在缺失的键，则报 warning
    if missing:
        print("WARNING {}: superfluous {}".format(_callable, missing), file=sys.stderr)

    # 将全部参数 merged 传给 _callable 并返回
    return _callable(**merged)
