# interactive
# Raymond Li, 2020-04-27
# Copyright (c) 2020 Element AI Inc. All rights reserved.


import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk

nltk.data.path.append('/home/chailz-s20/data/nltk_data')

import argparse
import glob
import re
import glob
import _jsonnet

import json
import os
import sqlite3
import re
import subprocess
from typing import Optional

import torch
import traceback

from duorat.datasets.spider import (
    SpiderItem,
    load_tables,
    SpiderSchema,
    schema_dict_to_spider_schema,
)
from duorat.preproc.utils import preprocess_schema_uncached
from duorat.types import RATPreprocItem, SQLSchema, Dict
from third_party.spider.preprocess.get_tables import dump_db_json_schema

from duorat.utils import saver as saver_mod

from duorat.asdl.lang.dusql.spider_transition_system import SpiderTransitionSystem

from duorat.api import DuoratAPI
from duorat.utils.evaluation import find_any_config
from duorat.preproc.slml import pretty_format_slml


def _process_name(name: str):
    # camelCase to spaces
    name = re.sub("([a-z])([A-Z])", "\g<1> \g<2>", name)
    return name.replace("-", " ").replace("_", " ").lower()


def _prompt_table(table_name, prompt_user=False):
    table_name = _process_name(table_name)
    # print(f"Current table name: {table_name}")
    new_name = (
        input("Type new name (empty to keep previous name): ") if prompt_user else ""
    )
    return new_name if new_name != "" else table_name


def _prompt_column(column_name, table_name, prompt_user=False):
    column_name = _process_name(column_name)
    # print(f"Table {table_name}. Current col name: {column_name}")
    new_name = (
        input("Type new name (empty to keep previous name): ") if prompt_user else ""
    )
    return new_name if new_name != "" else column_name


def refine_schema_names(schema: Dict):
    new_schema = {
        "column_names": [],
        "column_names_original": schema["column_names_original"],
        "original_column_names": schema["column_names_original"],
        "column_types": schema["column_types"],
        "db_id": schema["db_id"],
        "foreign_keys": schema["foreign_keys"],
        "primary_keys": schema["primary_keys"],
        "table_names": [],
        "table_names_original": schema["table_names_original"],
        "original_table_names": schema["table_names_original"],
    }
    for table in schema["table_names_original"]:
        corrected = _prompt_table(table)
        new_schema["table_names"].append(corrected)
    for col in schema["column_names_original"]:
        t_id = col[0]
        column_name = col[1]
        corrected = _prompt_column(column_name, new_schema["table_names"][t_id])
        new_schema["column_names"].append([t_id, corrected])
    return new_schema


class DuoratClient(object):

    def __init__(self, logdir, config_path, step, database_path, schema_path):
        self.model = DuoratAPI(logdir, config_path, step)
        self.schema_dict = self.load_schema_dict(database_path, schema_path)

    def load_schema_dict(self, database_path, schema_path):
        schema_dict = {}
        schemas, _ = load_tables([schema_path])

        db_name_list = os.listdir(database_path)
        for db_name in db_name_list:
            schema_dict[db_name] = {}
            schema_dict[db_name]['db_path'] = os.path.join(database_path, db_name, db_name + '.db')

            schema = schemas[db_name]
            schema['column_names_original'] = schema['original_column_names']
            schema['table_names_original'] = schema['original_table_names']
            column_names = schema['column_names_original']
            for column in column_names:
                if column[0] != -1:
                    column[0] = int(column[0].split('_')[-1])

            schema: SpiderSchema = schema_dict_to_spider_schema(
                refine_schema_names(schema)
            )

            schema_dict[db_name]['schema'] = schema
            preprocessed_schema: SQLSchema = preprocess_schema_uncached(
                schema=schema,
                db_path=schema_dict[db_name]['db_path'],
                tokenize=self.model.preproc._schema_tokenize,
            )
            schema_dict[db_name]['preprocessed_schema'] = preprocessed_schema

        return schema_dict

    def ask_any_question(self, question):
        results = self.duorat.infer_query(question)

        # print(pretty_format_slml(results['slml_question']))
        print(f'{results["query"]}  ({results["score"]})')
        try:
            results = self.duorat.execute(results['query'])
            print(results)
        except Exception as e:
            print(str(e))

    def show_schema(self):
        for table in self.duorat.schema.tables:
            print("Table", f"{table.name} ({table.orig_name})")
            for column in table.columns:
                print("    Column", f"{column.name} ({column.orig_name})")

    def infer(self, question, db_name):

        schema = self.schema_dict[db_name]['schema']
        preprocessed_schema = self.schema_dict[db_name]['preprocessed_schema']
        db_path = self.schema_dict[db_name]['db_path']
        # print(schema)
        results = self.model.infer_query(question, schema, preprocessed_schema)
        print(results['query'])
        query = results['query']
        conn = sqlite3.connect(db_path)
        # Temporary Hack: makes sure all literals are collated in a case-insensitive way
        # query = add_collate_nocase(query)
        results = conn.execute(query).fetchall()
        conn.close()
        return {'sql_query': query, 'sql_results': results}

    def run(self):
        self.show_schema()

        while True:
            question = input("Ask a question: ")
            self.ask_any_question(question)


if __name__ == "__main__":
    config = './logdir/config-20201129T191537.json'
    logdir = './logdir'
    db_path = './data/V2.0/database'
    schema_path = 'data/V2.0/db_schema_2.0.json'
    print('start to load model')
    client = DuoratClient(logdir=logdir, config_path=config, step=50000,
                          database_path=db_path,
                          schema_path=schema_path
                          )
    question = '每种意义中，最低定价最高的3个意义，对应的报纸编辑单位有哪些？'
    db_name = '报纸'
    print(question)
    print('start to infer and query')
    results = client.infer(question, db_name)
    print(results)

    question = '统计日期是2018-08-14且运输仓储业信心指数不为880的英国商业信心季度信息的所有数据有哪些？'
    db_name = '世界经济景气指数_英国'
    print(question)
    results = client.infer(question, db_name)
    print(results)
