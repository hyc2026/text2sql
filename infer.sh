nohup python scripts/infer_questions.py --logdir ./logdir/dusql-pre-spdb-electra-base \
                                    --data-config data/spdb/val.libsonnet \
                                    --questions data/spdb/newSPDB/spdb_val_1.5.json \
                                    --output-spider ./logdir/dusql-pre-spdb-electra-base/val_step_50000.json \
                                    --step 50000 \
                                    --prefix \"data/spdb/\" >train.log 2>&1 &


