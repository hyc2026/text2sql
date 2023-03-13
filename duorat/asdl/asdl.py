# Copyright 2018 Pengcheng Yin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from dataclasses import dataclass
from functools import total_ordering
from itertools import chain
from typing import Tuple, Dict, List, Sequence, Set, Iterable

from .utils import remove_comment

# Production举例：
# from = from(table_unit* table_units, cond? conds)
# Production: type constructor
# constructor: name filed
# filed: name type numbertype
# Prod.type = Prod.constructor
#           = Cons.name(Cons.filed, Cons.filed,...)
#           = Cons.name(filed.type numbertype filed.name , Cons.filed,...)


class ASDLGrammar(object):
    """
    Collection of types, constructors and productions
    """

    def __init__(self, productions: Sequence["ASDLProduction"]):
        # productions are indexed by their head types
        # 类型到该类所有推导方式的映射
        self._productions: OrderedDict[ASDLType, List[ASDLProduction]] = OrderedDict()

        # constructor 到推导方式的映射（子节点到父节点）
        self._constructor_production_map: Dict[str, ASDLProduction] = dict()
        for prod in productions:
            if prod.type not in self._productions:
                self._productions[prod.type] = list()
            self._productions[prod.type].append(prod)
            # print(prod)
            self._constructor_production_map[prod.constructor.name] = prod

        # 根类型 sql
        self.root_type: ASDLType = productions[0].type
        # number of constructors 推导方式的总数
        self.size: int = sum(len(head) for head in self._productions.values())

        # get entities to their ids map
        self.prod2id: Dict[ASDLProduction, int] = {
            prod: i for i, prod in enumerate(self.productions)
        }
        # self.types 为 pord、field 中出现的所有类型
        self.type2id: Dict[ASDLType, int] = {
            type: i for i, type in enumerate(self.types)
        }
        # self.field 为所有子节点
        self.field2id: Dict[Field, int] = {
            field: i for i, field in enumerate(self.fields)
        }

        self.id2prod: Dict[int, ASDLProduction] = {
            i: prod for i, prod in enumerate(self.productions)
        }
        self.id2type: Dict[int, ASDLType] = {
            i: type for i, type in enumerate(self.types)
        }
        self.id2field: Dict[int, Field] = {
            i: field for i, field in enumerate(self.fields)
        }

    def __len__(self) -> int:
        return self.size

    @property
    def productions(self) -> List["ASDLProduction"]:
        return sorted(
            chain.from_iterable(self._productions.values()), key=lambda x: repr(x)
        )

    def __getitem__(self, type: "ASDLType") -> List["ASDLProduction"]:
        return self._productions[type]

    def get_prod_by_ctr_name(self, name: str) -> "ASDLProduction":
        return self._constructor_production_map[name]

    @property
    def types(self) -> List["ASDLType"]:
        if not hasattr(self, "_types"):
            all_types: Set[ASDLType] = set()
            for prod in self.productions:
                all_types.add(prod.type)
                all_types.update(map(lambda x: x.type, prod.constructor.fields))

            self._types: List[ASDLType] = sorted(all_types, key=lambda x: x.name)

        return self._types

    @property
    def fields(self) -> List["Field"]:
        if not hasattr(self, "_fields"):
            all_fields: Set[Field] = set()
            for prod in self.productions:
                all_fields.update(prod.constructor.fields)

            self._fields: List[Field] = sorted(
                all_fields, key=lambda x: (x.name, x.type.name, x.cardinality)
            )

        return self._fields

    @property
    def primitive_types(self) -> Iterable["ASDLPrimitiveType"]:
        return filter(lambda x: isinstance(x, ASDLPrimitiveType), self.types)

    @property
    def composite_types(self) -> Iterable["ASDLCompositeType"]:
        return filter(lambda x: isinstance(x, ASDLCompositeType), self.types)

    def is_composite_type(self, asdl_type: "ASDLType") -> bool:
        """是不是一个非叶子节点"""
        return asdl_type in self.composite_types

    def is_primitive_type(self, asdl_type: "ASDLType") -> bool:
        """是不是一个叶子节点"""
        return asdl_type in self.primitive_types

    @staticmethod
    def from_text(text: str) -> "ASDLGrammar":
        def _parse_field_from_text(_text):
            # col_unit col_unit1
            # 形式为 "类型 名称"
            # 类似于正则表达式 *表示零个或多个，?表示一个或零个
            # e.g. from 的语法 from = from(table_unit* table_units, cond? conds)
            d = _text.strip().split(" ")
            name = d[1].strip()
            type_str = d[0].strip()
            cardinality = "single"
            if type_str[-1] == "*":
                type_str = type_str[:-1]
                cardinality = "multiple"
            elif type_str[-1] == "?":
                type_str = type_str[:-1]
                cardinality = "optional"

            if type_str in primitive_type_names:
                return Field(name, ASDLPrimitiveType(type_str), cardinality=cardinality)
            else:
                return Field(name, ASDLCompositeType(type_str), cardinality=cardinality)

        def _parse_constructor_from_text(_text):
            _text = _text.strip()
            fields = None
            if "(" in _text:
                # 如果有'('，说明括号前的部分是元素名称
                # 比如 Column(col_unit col_unit1)
                name = _text[: _text.find("(")]

                # 子元素块，由','分隔后解析
                field_blocks = _text[_text.find("(") + 1 : _text.find(")")].split(",")
                fields = map(_parse_field_from_text, field_blocks)
            else:
                name = _text

            if name == "":
                name = None

            return ASDLConstructor(
                name, tuple(fields) if fields is not None else tuple()
            )

        # 读取语法文件所有非 # 注释的行，strip 后去掉首尾空白
        lines = remove_comment(text).split("\n")
        lines = list(map(lambda l: l.strip(), lines))
        lines = list(filter(lambda l: l, lines))
        line_no = 0

        # first line is always the primitive types
        # 文件中第一行记录了原子类型
        # spider 中为 [string, column, singleton, object, int, table]
        primitive_type_names = list(map(lambda x: x.strip(), lines[line_no].split(",")))
        line_no += 1

        all_productions = list()

        while True:
            # 从文件中读取一个新的语法类型，形式为 type_name = constructors_block | constructors_block  ...
            type_block = lines[line_no]
            type_name = type_block[: type_block.find("=")].strip()
            constructors_blocks = type_block[type_block.find("=") + 1 :].split("|")
            i = line_no + 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                t = lines[i].strip()
                cont_constructors_blocks = t[1:].split("|")
                constructors_blocks.extend(cont_constructors_blocks)

                i += 1

            constructors_blocks = filter(lambda x: x and x.strip(), constructors_blocks)

            # parse type name
            # 判断该语法类型是不是原子类型
            new_type = (
                ASDLPrimitiveType(type_name)
                if type_name in primitive_type_names
                else ASDLCompositeType(type_name)
            )
            # 解析该语法类型的构成元素
            constructors = map(_parse_constructor_from_text, constructors_blocks)

            # 记录推导式
            productions = list(map(lambda c: ASDLProduction(new_type, c), constructors))
            all_productions.extend(productions)

            line_no = i
            if line_no == len(lines):
                break

        grammar = ASDLGrammar(all_productions)

        return grammar


class ASDLType(object):
    name: str

    def __repr__(self, plain: bool = False) -> str:
        raise NotImplementedError


"""
@dataclass 为数据类修饰器，将自动为类添加 init 函数

比如对于 ASDLCompositeType，自动添加函数
def __init__(self, name: str):
    self.name = name 
"""


@dataclass(order=False, eq=False, unsafe_hash=True, frozen=True)
@total_ordering
class ASDLCompositeType(ASDLType):
    name: str

    def __repr__(self, plain: bool = False) -> str:
        plain_repr = self.name
        if plain:
            return plain_repr
        else:
            return "%s(%s)" % (self.__class__.__name__, plain_repr)

    def __eq__(self, other: "ASDLType"):
        if isinstance(other, ASDLCompositeType):
            return self.name == other.name
        elif isinstance(other, ASDLType):
            return False
        else:
            return NotImplemented

    def __lt__(self, other: "ASDLType") -> bool:
        if isinstance(other, ASDLCompositeType):
            return self.name < other.name
        elif isinstance(other, ASDLPrimitiveType):
            return True
        else:
            return NotImplemented


@dataclass(order=False, eq=False, unsafe_hash=True, frozen=True)
@total_ordering
class ASDLPrimitiveType(ASDLType):
    name: str

    def __repr__(self, plain: bool = False) -> str:
        plain_repr = self.name
        if plain:
            return plain_repr
        else:
            return "%s(%s)" % (self.__class__.__name__, plain_repr)

    def __eq__(self, other: "ASDLType"):
        if isinstance(other, ASDLPrimitiveType):
            return self.name == other.name
        elif isinstance(other, ASDLType):
            return False
        else:
            return NotImplemented

    def __lt__(self, other: "ASDLType") -> bool:
        if isinstance(other, ASDLCompositeType):
            return False
        elif isinstance(other, ASDLPrimitiveType):
            return self.name < other.name
        else:
            return NotImplemented


@dataclass(order=True, frozen=True)
class Field(object):
    name: str
    type: ASDLType
    cardinality: str

    def __repr__(self, plain: bool = False) -> str:
        plain_repr = "%s%s %s" % (
            self.type.__repr__(plain=True),
            Field._get_cardinality_repr(self.cardinality),
            self.name,
        )
        if plain:
            return plain_repr
        else:
            return "Field(%s)" % plain_repr

    @staticmethod
    def _get_cardinality_repr(cardinality: str) -> str:
        return (
            "" if cardinality == "single" else "?" if cardinality == "optional" else "*"
        )


@dataclass(order=True, frozen=True)
class ASDLConstructor(object):
    name: str
    fields: Tuple[Field, ...]

    def __getitem__(self, field_name: str) -> Field:
        for field in self.fields:
            if field.name == field_name:
                return field

        raise KeyError

    def __repr__(self, plain=False):
        plain_repr = "%s(%s)" % (
            self.name,
            ", ".join(f.__repr__(plain=True) for f in self.fields),
        )
        if plain:
            return plain_repr
        else:
            return "Constructor(%s)" % plain_repr


@dataclass(order=True, frozen=True)
class ASDLProduction(object):
    type: ASDLType
    constructor: ASDLConstructor

    @property
    def fields(self) -> Tuple[Field, ...]:
        return self.constructor.fields

    def __getitem__(self, field_name) -> Field:
        return self.constructor[field_name]

    def __repr__(self):
        return "%s -> %s" % (
            self.type.__repr__(plain=True),
            self.constructor.__repr__(plain=True),
        )
