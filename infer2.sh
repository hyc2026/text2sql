python scripts/infer_questions_multi_proc.py \
                                    --logdir ./logdir/dusql-pre-spdb-electra-base \
                                    --data-config data/spdb/val.libsonnet \
                                    --questions data/spdb/newSPDB/spdb_val_1.5.json \
                                    --output-spider ./logdir/dusql-pre-spdb-electra-base/val_step_mult.json \
                                    --step 50000 \
                                    --nproc 4 \
                                    --prefix \"data/spdb/\" 


