export CUDA_VISIBLE_DEVICES=2
nohup python -u scripts/train.py \
        --config configs/duorat/dusql-electra-base.jsonnet \
        --logdir logdir/duorat-electra-base-2.0-fold3-6layer \
        --step 30000 >train_fold3.log 2>&1 &
