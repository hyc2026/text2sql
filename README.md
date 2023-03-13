# DuoRAT
# DuoRAT

## 目录介绍

```
├── configs
│   ├── duorat # 模型配置文件
│   │   └── ...
├── data
│   ├── configs # 数据配置文件，主要内容包括涉及到的dataset及其路径
│   │   └── ...
│   ├── test_data # ，包含schema,content,test的json文件
│   └── test_dataset # test data 按数据库split后的结果
│   └── train_data # 训练数据schema,content,test的json文件
│   └── train_dataset #train data 按数据库split后的结果
├── logdir # 保存的模型文件路径
├── pretrain # 预训练模型路径，比如bert，roberta等
├── duorat # 这里面是核心代码文件
├── scripts # 存放了直接可以运行的各个.py，如train.py等
├── requirements.txt # 核心依赖包
└── infer.sh train.sh infer2.sh # 常用的.sh文件，完成infer和train工作
```



## 环境配置

- os: Ubuntu 18.04
- python: 3.7
- CUDA: 10.0
- cuda: 9.2
- cudnn: 7.6.5

本环境使用`conda`创建的虚拟环境。

```bash
conda create -n duorat python=3.7
```

安装完毕后，先激活环境：

```bash
conda activate duorat
```

安装`cudatoolkit`和`pytorch`命令：

```bash
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=9.2 -c pytorch
```

然后切换到项目目录，安装依赖包：

```bash
pip install -r requirements.txt
```

## 数据

- dusql 数据格式
- 训练集：train.json；db_schema.json；db_content.json
- 测试集：test.json；db_schema.json；db_content.json



## 工作流程


### 运行前准备

通过`conda activate duorat`激活环境，并进入`~/data/duo-rat-2020`文件夹下

#### train
  前置依赖：已有处理好的，转换为dusql形式的训练集与验证集
  流程：

1. 运行`scripts/split_dusql_bydb.py`，得到按数据库划分后的训练数据和验证数据， 其中需要修改内容使之同data下的文件命名和路径一致
```python
# 修改10行开始的内容，使之命名一致
tables_json_path = "newSPDB_db_schema.json"
content_json_path = 'newSPDB_db_content.json'
examples_paths = ["spdb_train.json", "spdb_val.json"]

#修改86行开始内容，指定dataset路径和split后结果的存放路径

parser.add_argument("--data-path", type=str, default='./data/newSPDB')
parser.add_argument("--duorat-path", type=str, default='./data/database')  
```


2. 创建data config，将前一步输出的`train`数据依赖数据库名和`val`数据依赖数据库名加入到对应的`train.libsonnet`和`val.libsonnet`中
   
3. 创建model condig，在其中配置模型和训练过程的各项参数，并在data部分指定第2步配置好的`train.libsonnet`和`val.libsonnet`
   
4. 修改`train.sh`，配置logdir等参数
   
5. `sh train.sh`即可，train.sh内容如下
        
```bash
nohup python -u scripts/train.py \
    --config configs/duorat/dusql-electra-base.jsonnet \
    --logdir logdir/3.0-for-hornor  >train_best.log 2>&1 &
tail -f train_best.log
```

#### infer
 前置依赖：已有处理好的，转换为dusql形式的测试集
 流程：

1. 运行`scripts/split_test.py`，得到按数据库划分后的测试数据
2. 创建data config，将前一步输出的`test`数据依赖数据库名和加入到对应的`test.libsonnet`
3. 修改 infer2.sh，指定logdir和datadir
`infer2.sh`内容
```bash
nohup python scripts/infer_questions_multi_proc.py --logdir ./logdir/3.0-for-hornor \
                --data-config data/test.libsonnet \
                --questions data/test_split/test.json \
                --output-spider ./logdir/3.0-for-hornor/infer_35000.json \
                --nproc 4 \
                --step 35000  >infer_3.0_35000.log 2>&1 &
tail -f infer_3.0_35000.log
```



## 参考资料

- 论文：[DuoRAT: Towards Simpler Text-to-SQL Models](https://arxiv.org/pdf/2010.11119.pdf)
- 代码：[https://github.com/ElementAI/duorat](https://github.com/ElementAI/duorat)


