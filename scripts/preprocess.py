import argparse
import json
import _jsonnet
import tqdm

# noinspection PyUnresolvedReferences
from duorat import datasets

# noinspection PyUnresolvedReferences
from duorat.preproc import offline, utils

# noinspection PyUnresolvedReferences
from duorat.utils import schema_linker

# noinspection PyUnresolvedReferences
from duorat.asdl.lang import dusql as spider

from duorat.utils import registry


class Preprocessor:
    def __init__(self, config):
        self.config = config

        # 根据参数 config 构造预处理器，self.model_preproc 为 BertDuoRAT
        # 参考 duorat/preproc/offline.py 的类 BertDuoRATPreproc 及其基类 DuoRATPreproc
        self.model_preproc = registry.construct(
            "preproc", self.config["model"]["preproc"],
        )

    def preprocess(self, sections, keep_vocab):
        """
        :param sections: 为 ["train", "val"]
        :param keep_vocab: 是否保留词表
        :return:
        """
        # clear_items 函数 重置字典 self.preproc_items: Dict[str, List[RATPreprocItem]] = {}
        self.model_preproc.clear_items()

        # assert sections == ["train", "val"]
        for section in sections:
            # 分别根据配置文件构建 训练集 和 验证集
            # 构造的 data 对象属于 duorat/datasets/spider.py 中的类 SpiderDataset
            data = registry.construct("dataset", self.config["data"][section])
            for item in tqdm.tqdm(data, desc=section, dynamic_ncols=True):
                # 逐个验证数据，如果合法 则 to_add 返回 True，而 validation_info 返回 asdl_ast
                to_add, validation_info = self.model_preproc.validate_item(
                    item, section
                )
                # 如果合法，则对 item 进行预处理后 加入到 vocab 中
                if to_add:
                    self.model_preproc.add_item(item, section, validation_info)

        # TODO (YuweiYin) 疑问：这里是不是写反了，看代码实现里 save 函数才是真的存储了 vocab，而 save_examples 仅存储数据
        if keep_vocab:
            # save_examples 函数根据 "save_path" 参数创建目录，并把 train 和 val 数据分别存储于该目录下的 .pkl 文件中
            self.model_preproc.save_examples()
        else:
            # save 函数一开始就会调用 save_examples 函数
            # 之后 production rules + Reduce + MASK + GenToken tokens that are *not* in the encoder sequence
            # 最后存储 self.target_vocab
            self.model_preproc.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-args")
    parser.add_argument("--sections", nargs='+', default=None,
                        help="Preprocess only the listed sections")
    parser.add_argument("--keep-vocab", action='store_true',
                        help="Keep existing vocabulary files")
    args = parser.parse_args()

    # 此后关于 config 的叙述均根据配置文件 duorat-finetune-bert-large.jsonnet

    if args.config_args:
        config = json.loads(
            _jsonnet.evaluate_file(args.config, tla_codes={"args": args.config_args})
        )
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    # 结果为 sections = ["train", "val"]
    sections = args.sections if args.sections is not None else config["data"].keys()

    preprocessor = Preprocessor(config)
    preprocessor.preprocess(sections, args.keep_vocab)


if __name__ == "__main__":
    main()
