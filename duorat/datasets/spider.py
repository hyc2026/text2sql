import os
import dataclasses
import json
from typing import Optional, Tuple, List, Iterable

import attr
import networkx as nx
from networkx.drawing.layout import shell_layout
from pydantic.dataclasses import dataclass
from pydantic.main import BaseConfig
from torch.utils.data import Dataset

from duorat.utils import registry
from third_party.spider import evaluation
from third_party.spider.preprocess.schema import get_schemas_from_json, Schema
from third_party.spider.process_sql import get_sql


@dataclass
class SpiderTable:
    """数据表"""
    id: int  # 表号
    name: List[str]  # 表名 list
    unsplit_name: str  # TODO (YuweiYin) 解释此属性含义/作用
    orig_name: str  # TODO (YuweiYin) 解释此属性含义/作用
    orig_name_for_unparse: str # 用于在unparse时输出原始表名
    columns: List["SpiderColumn"] = dataclasses.field(default_factory=list)  # 本表含有的列 list
    primary_keys: List[str] = dataclasses.field(default_factory=list)  # 本表的主键 list


@dataclass
class SpiderColumn:
    """数据列"""
    id: int  # 列号
    table: Optional[SpiderTable]  # 本列属于那个表
    name: List[str]  # 列名 list
    unsplit_name: str  # TODO (YuweiYin) 解释此属性含义/作用
    orig_name: str  # TODO (YuweiYin) 解释此属性含义/作用
    orig_name_for_unparse: str # 用于在unparse时输出原始列名
    type: str  # 列类型名称
    foreign_key_for: Optional[str] = None  # 外键名称


SpiderTable.__pydantic_model__.update_forward_refs()


class SpiderSchemaConfig:
    arbitrary_types_allowed = True


@dataclass(config=SpiderSchemaConfig)
class SpiderSchema(BaseConfig):
    """数据库"""
    db_id: str  # 数据库号
    tables: Tuple[SpiderTable, ...]  # 本数据库含有的数据表 list
    columns: Tuple[SpiderColumn, ...]  # 本数据库含有的数据列 list
    foreign_key_graph: nx.DiGraph  # 通过外键关系构建的有向图
    orig: dict  # TODO (YuweiYin) 解释此属性含义/作用


@dataclass
class SpiderItem:
    """数据项 slml: Schema Linking Markup Language"""
    question: str  # 自然语言问题
    slml_question: Optional[str]  # TODO (YuweiYin) 解释此属性含义/作用 (应该是对 question 进行 parsing 后的 AST 相关结构)
    query: str  # SQL 语句 (真值标签)
    spider_sql: dict  # 数据集中给出的 sql 结构
    spider_schema: SpiderSchema  # 本数据项所在数据库
    db_path: str  # 本数据项所在数据库原始文件的路径
    orig: dict  # TODO (YuweiYin) 解释此属性含义/作用

def schema_dict_to_spider_schema(schema_dict):
    # 建立各个数据表 SpiderTable 对象，构成元组
    tables = tuple(
        SpiderTable(id=i, name=name.split(), unsplit_name=name, orig_name=orig_name, orig_name_for_unparse=orig_name_for_unparse)
        for i, (name, orig_name, orig_name_for_unparse) in enumerate(
            zip(schema_dict["table_names"], schema_dict["table_names_original"], schema_dict["original_table_names"])
        )
    )
    # 建立各个数据列 SpiderColumn 对象，构成元组
    columns = tuple(
        SpiderColumn(
            id=i,
            table=tables[table_id] if table_id >= 0 else None,
            name=col_name.split(),
            unsplit_name=col_name,
            orig_name=orig_col_name,
            orig_name_for_unparse=orig_col_name_for_unparse,
            type=col_type,
        )
        for i, ((table_id, col_name), (_, orig_col_name), col_type, (_, orig_col_name_for_unparse)) in enumerate(
            zip(
                schema_dict["column_names"],
                schema_dict["column_names_original"],
                schema_dict["column_types"],
                schema_dict["original_column_names"]
            )
        )
    )

    # Link columns to tables 让数据列链指向 其所在的数据表
    for column in columns:
        if column.table:
            column.table.columns.append(column)

    # 设置数据列的主键
    for column_id in schema_dict["primary_keys"]:
        # Register primary keys
        column = columns[column_id]
        column.table.primary_keys.append(column)

    # 设置外键图结构，有向边为 源列 和 目标列 的双向边
    foreign_key_graph = nx.DiGraph()
    for source_column_id, dest_column_id in schema_dict["foreign_keys"]:
        # Register foreign keys
        source_column = columns[source_column_id]
        dest_column = columns[dest_column_id]
        source_column.foreign_key_for = dest_column

        # 设置双向链接
        foreign_key_graph.add_edge(
            source_column.table.id,
            dest_column.table.id,
            columns=(source_column_id, dest_column_id),
        )
        foreign_key_graph.add_edge(
            dest_column.table.id,
            source_column.table.id,
            columns=(dest_column_id, source_column_id),
        )

    # 建立并返回数据库 SpiderSchema 对象
    db_id = schema_dict["db_id"]
    return SpiderSchema(db_id, tables, columns, foreign_key_graph, schema_dict)


def load_tables(paths):
    """
    :param paths: 列表，每个列表项为某个数据库的目录路径
    :return: 全部数据库 和 外键映射字典
    """
    schemas = {}
    eval_foreign_key_maps = {}

    for path in paths:
        schema_dicts = json.load(open(path))
        for schema_dict in schema_dicts:
            db_id = schema_dict["db_id"]
            assert db_id not in schemas
            
            schemas[db_id] = schema_dict
            eval_foreign_key_maps[db_id] = evaluation.build_foreign_key_map(schema_dict)

    return schemas, eval_foreign_key_maps


def load_original_schemas(tables_paths):
    # 从 json 文件加载原始数据库结构
    all_schemas = {}
    for path in tables_paths:
        schemas, db_ids, tables = get_schemas_from_json(path)
        for db_id in db_ids:
            all_schemas[db_id] = Schema(schemas[db_id], tables[db_id])
    return all_schemas


def check_now_time(entry, schema):
    """
    检查 entry 数据项中的 TIME_NOW，将其转化为表中一列
    """
    if schema['column_names'][-1] != [0, '当前时间']:
        schema['column_names'].append([0, '当前时间'])
        schema['column_types'].append('time')
        schema['column_names_original'].append([0, '当前时间'])
        schema['original_column_names'].append([0, '当前时间'])
    #print(entry['sql'])
    if 'where' in entry['sql'].keys():
        where_cond = entry['sql']['where']
        if len(where_cond) > 0:
            temp = where_cond[0][2][1][1]
            if temp == 'TIME_NOW':
                entry['sql']['where'][0][2][1][1] = len(schema['column_names'])-1
                return True
    return False


@registry.register("dataset", "spider")
class SpiderDataset(Dataset):
    def __init__(self, paths: List[str], tables_paths: List[str], db_path: str):
        self.paths = paths
        self.db_path = db_path
        self.examples = []  # 全部数据项对象 list

        # 加载全部数据库 和 外键映射字典
        self.schemas, self.eval_foreign_key_maps = load_tables(tables_paths)

        raw_datas = []
        # 处理 time_now 的 column
        for path in paths:
            # 读入数据
            raw_data = json.load(open(path))
            for i, entry in enumerate(raw_data):
                # 检查 entry 数据项中的 TIME_NOW 列，将其转化为表中一列
                check_now_time(entry, self.schemas[entry['db_id']])
            
            # 重写数据会引起bug, 直接把修改后的数据放到数组里
            raw_datas.append(raw_data)

        # 建立各个数据库对象 (及其内部的数据表、数据列对象)
        for db_id, schema_dict in self.schemas.items():
            self.schemas[db_id] = schema_dict_to_spider_schema(schema_dict)

        # 从 json 文件加载原始数据库结构
        original_schemas = load_original_schemas(tables_paths)

        # 遍历原始数据
        for raw_data in raw_datas:    
            for entry in raw_data:
                # 获得原始数据的 sql 项，它用于构成 AST 抽象语法树
                if "sql" not in entry:
                    entry["sql"] = get_sql(
                        original_schemas[entry["db_id"]], entry["query"]
                    )
                # 构建数据项对象
                item = SpiderItem(
                    question=entry["question"],
                    slml_question=entry.get("slml_question", None),
                    query=entry["original_sql_query"] if 'original_sql_query' in entry.keys() else entry["query"],
                    spider_sql=entry["sql"],
                    spider_schema=self.schemas[entry["db_id"]],
                    db_path=self.get_db_path(entry["db_id"]),
                    orig=entry,
                )
                # 将此数据项加入 self.examples 中，即全部数据项对象 list
                self.examples.append(item)

    def get_db_path(self, db_id: str):
        return os.path.join(self.db_path, db_id, 'contents.json')

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> SpiderItem:
        return self.examples[idx]

    class Metrics:
        def __init__(self, dataset):
            self.dataset = dataset
            self.foreign_key_maps = {
                db_id: evaluation.build_foreign_key_map(schema.orig)
                for db_id, schema in self.dataset.schemas.items()
            }
            self.evaluator = evaluation.Evaluator(
                self.dataset.db_path, self.foreign_key_maps, "match"
            )
            self.results = []

        def add(self, item: SpiderItem, inferred_code: str):
            res = self.evaluator.evaluate_one(
                    db_name=item.spider_schema.db_id,
                    gold=item.query,
                    predicted=inferred_code,
                )
            self.results.append(res)
            return res 

        def evaluate_all(
            self, idx: int, item: SpiderItem, inferred_codes: Iterable[str]
        ) -> Tuple[int, list]:
            beams = [
                self.evaluator.evaluate_one(
                    db_name=item.spider_schema.db_id,
                    gold=item.query,
                    predicted=inferred_code,
                )
                for inferred_code in inferred_codes
            ]
            return idx, beams

        def finalize(self) -> dict:
            self.evaluator.finalize()
            return {
                "per_item": self.results, 
                "total_scores": self.evaluator.scores,
                }
