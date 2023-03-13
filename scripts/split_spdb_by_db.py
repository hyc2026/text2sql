import json
import os
import argparse

from collections import defaultdict
from typing import List, Dict


def main(data_path, duorat_path) -> None:
      
    tables_json_path = "newSPDB_db_schema_1.5.json"
    content_json_path = 'newSPDB_db_content_1.5.json'
    examples_paths = ["spdb_train_1.5.json", "spdb_val_1.5.json"]

    ### 1. Produce tables.json files
    with open(os.path.join(data_path, tables_json_path), "r") as read_fp:
        payload: List[dict] = json.load(read_fp)

    grouped_payload: Dict[str, dict] = {}
    for item in payload:
        db_id: str = item['db_id']
        db_id_new = db_id.replace('/','_')
        if db_id_new!=db_id:
            print('rename',db_id, db_id_new)
            item['db_id'] = db_id_new
        assert db_id_new not in grouped_payload
        grouped_payload[db_id_new] = item

    for db_id, item in grouped_payload.items():  
        item['column_names_original'] = item['column_names']
        item['table_names_original'] = item['table_names']   
        db_dir = os.path.join(duorat_path, db_id)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        with open(os.path.join(db_dir, "tables.json"), "wt",encoding='utf-8') as write_fp:
            json.dump([item], write_fp, indent=2, ensure_ascii=False)
    
    with open(os.path.join(data_path, "new_db_schema.json"), "w") as write_fp:
        json.dump(payload, write_fp, indent=2, ensure_ascii=False)
    ### 2. Produce examples.json files
    for examples_path in examples_paths:
        
        db_set = []
        with open(os.path.join(data_path, examples_path), "r") as read_fp:
            payload: List[dict] = json.load(read_fp)
        

        grouped_payload: Dict[str, List[dict]] = defaultdict(list)
        for item in payload:
            item['query'] = item.pop('sql_query')

            db_id: str = item['db_id']
            db_id_new = db_id.replace('/','_')
            if db_id_new!=db_id:
                print('rename',db_id, db_id_new)
                item['db_id'] = db_id_new
    
            grouped_payload[db_id_new].append(item)
            db_set.append(db_id_new)    
        for db_id, payload_group in grouped_payload.items():
            
            with open(os.path.join(duorat_path, db_id, "examples.json"), "wt") as write_fp:
                json.dump(payload_group, write_fp, indent=2, ensure_ascii=False)
        print(examples_path, set(db_set))
        with open(os.path.join(data_path, "new_" + examples_path), "w") as write_fp:
            json.dump(payload, write_fp, indent=2, ensure_ascii=False)
    ### 3. Produce content.json files
    with open(os.path.join(data_path, content_json_path), "r") as read_fp:
        payload: List[dict] = json.load(read_fp)
    grouped_payload: Dict[str, dict] = {}
    for item in payload:
        db_id: str = item['db_id']
        db_id_new = db_id.replace('/','_')
        if db_id_new!=db_id:
            print('rename',db_id, db_id_new)
            item['db_id'] = db_id_new
        assert db_id_new not in grouped_payload
        grouped_payload[db_id_new] = item

    for db_id, item in grouped_payload.items():     
        db_dir = os.path.join(duorat_path, db_id)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        with open(os.path.join(db_dir, "contents.json"), "wt") as write_fp:
            json.dump([item], write_fp, indent=2, ensure_ascii=False) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='./data/spdb/newSPDB')
    parser.add_argument("--duorat-path", type=str, default='./data/spdb/database')
    args = parser.parse_args()

    main(args.data_path, args.duorat_path)
