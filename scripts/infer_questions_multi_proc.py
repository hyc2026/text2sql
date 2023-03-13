
import nltk
nltk.data.path.append('/home/guozl-s20/data/nltk_data')

import json
import os,sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import argparse
import functools
import _jsonnet
import tqdm
import torch 
import time 

from duorat.asdl.asdl_ast import AbstractSyntaxTree
from duorat.types import RATPreprocItem
from duorat.utils import registry, parallelizer
from duorat.api import DuoratAPI, DuoratOnDatabase
from duorat.utils.evaluation import find_any_config
from duorat.preproc.utils import preprocess_schema_uncached
from duorat.datasets.spider import SpiderDataset, SpiderItem
from duorat.preproc.slml import pretty_format_slml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Infer queries for questions about a data section. The output format"\
        " is compatible with official Spider eval scripts.")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--config",
        help="The configuration file. By default, an arbitrary configuration from the logdir is loaded")
    parser.add_argument("--data-config",
        help="Dataset section configuration",
        required=True)
    parser.add_argument(
        "--questions",
        help="The path to the questions in Spider format."
             "By default, use questions specified by --data-config",
    )
    parser.add_argument(
        "--output-spider",
        help="Path to save outputs in the Spider format")
    parser.add_argument(
        "--output-google",
        help="Path to save output in the Google format")

    parser.add_argument(
        "--step")    
    parser.add_argument(
        "--prefix", default='"data/"'
    )  
    parser.add_argument(
        "--nproc", default=1
    )

    args = parser.parse_args()

    if args.output_spider is None and args.output_google is None:
        raise ValueError("specify output destination in either Google or Michigan format")

    config_path = find_any_config(args.logdir) if args.config is None else args.config
    api = DuoratAPI(args.logdir, config_path, args.step)

    data_config = json.loads(_jsonnet.evaluate_file(
        args.data_config, tla_codes={'prefix': args.prefix}))
    if data_config['name'] != 'spider':
        raise ValueError()
    del data_config['name']
    if args.questions:
        data_config['paths'] = [args.questions]
    dataset = SpiderDataset(**data_config)

    sql_schemas = {}
    for db_id in dataset.schemas:
        spider_schema = dataset.schemas[db_id]
        sql_schemas[db_id] = preprocess_schema_uncached(
            schema=spider_schema,
            db_path=dataset.get_db_path(db_id),
            tokenize=api.preproc._schema_tokenize,
        )

    if args.output_spider and os.path.exists(args.output_spider):
        os.remove(args.output_spider)

    output_items = []

    orig_data = []
    preproc_data = []

    print('preprocessing data...')
    for item in tqdm.tqdm(dataset):
        db_id = item.spider_schema.db_id

        spider_item = SpiderItem(
            question=item.question,
            slml_question=None,
            query="",
            spider_sql={},
            spider_schema=item.spider_schema,
            db_path="",
            orig={'question_id':item.orig['question_id']},
        )
        preproc_item: RATPreprocItem = api.preproc.preprocess_item(
            spider_item,
            sql_schemas[db_id],
            AbstractSyntaxTree(production=None, fields=(), created_time=None),
        )
        orig_data.append(spider_item)
        preproc_data.append(preproc_item)

    print('start infer...')
    if torch.cuda.is_available():
        cp = parallelizer.CUDAParallelizer(int(args.nproc))
    else:
        cp = parallelizer.CPUParallelizer(int(args.nproc))
    inferred_lines = cp.parallel_map(
        [
            (
                functools.partial(
                    DuoratAPI.parse_single2,
                    api.model,
                    beam_size=1,
                    decode_max_time_step=500,
                ),
                list(enumerate(zip(orig_data, preproc_data))),
            )
        ]
    )
    inferred_lines = list(inferred_lines)

    with open(args.output_spider, "w") as output_dst:
        for line in inferred_lines:
         #   print(line)
            output_dst.write(line)

