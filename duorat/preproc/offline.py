import itertools
import os
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Optional, Set, Union
from uuid import uuid4

from dataclasses import replace
from torchtext.vocab import Vocab, GloVe, Vectors

from duorat.datasets.spider import SpiderItem
from duorat.asdl.transition_system import (
    MaskAction,
    ApplyRuleAction,
    ReduceAction,
    TransitionSystem,
)
from duorat.asdl.asdl_ast import AbstractSyntaxTree
from duorat.asdl.lang.dusql.spider_transition_system import (
    all_spider_gen_token_actions,
    SpiderTransitionSystem,
)
from duorat.preproc import abstract_preproc
from duorat.preproc.slml import SLMLParser
from duorat.preproc.utils import (
    ActionVocab,
    preprocess_schema_uncached,
    shuffle_schema,
)
from duorat.types import (
    RATPreprocItem,
    PreprocQuestionToken,
    SQLSchema,
    QuestionTokenId,
)
from duorat.utils import registry
from duorat.utils.schema_linker import AbstractSchemaLinker
from duorat.utils.tokenization import AbstractTokenizer, BERTTokenizer


class DuoRATPreproc(abstract_preproc.AbstractPreproc):
    def __init__(self, **kwargs) -> None:
        self.save_path = kwargs["save_path"]
        self.min_freq = kwargs["min_freq"]
        # Number of schema-shuffles, for data augmentation. 0 means no shuffling.
        # 默认配置中 train_num_schema_shuffles: 0, val_num_schema_shuffles: 0
        self.train_num_schema_shuffles = kwargs.get("train_num_schema_shuffles", 0)
        self.val_num_schema_shuffles = kwargs.get("val_num_schema_shuffles", 0)

        # for production rules + ReduceAction + MaskAction + GenToken tokens
        self.target_vocab_counter = Counter()
        self.target_vocab_path = os.path.join(self.save_path, "target_vocab.pkl")
        self.target_vocab = None

        self.counted_db_ids: Set[int] = set()
        self.sql_schemas: Dict[str, SQLSchema] = {}

        # 构建 duorat/utils/tokenization.py 文件下的分词器类，例如 BERTTokenizer 的对象
        self.tokenizer: AbstractTokenizer = registry.construct(
            "tokenizer", kwargs["tokenizer"]
        )

        # 构建 duorat/asdl/lang/dusql/spider_transition_system.py 中的类 SpiderTransitionSystem 对象
        # 类 SpiderTransitionSystem 的基类为 duorat/asdl/transition_system.py 中的类 TransitionSystem
        # 默认配置的相关参数为 transition_system+: { tokenizer: {
        #     name: 'BERTTokenizer',
        #     pretrained_model_name_or_path: 'bert-large-uncased-whole-word-masking',}}
        self.transition_system: TransitionSystem = registry.construct(
            "transition_system", kwargs["transition_system"]
        )

        # 构建 duorat/utils/schema_linker.py 中的类 SpiderSchemaLinker 对象
        # 它继承自同文件下的类 AbstractSchemaLinker(metaclass=abc.ABCMeta)
        self.schema_linker: AbstractSchemaLinker = registry.construct(
            "schema_linker", kwargs["schema_linker"]
        )

        # 处理后的对象字典
        self.preproc_items: Dict[str, List[RATPreprocItem]] = {}

    def input_a_str_to_id(self, s: str) -> int:
        raise NotImplementedError

    def input_b_str_to_id(self, s: str) -> int:
        raise NotImplementedError

    def _schema_tokenize(
        self, type: Optional[str], something: List[str], name: str
    ) -> List[str]:
        raise NotImplementedError

    def validate_item(
        self, item: SpiderItem, section: str
    ) -> Tuple[bool, Optional[AbstractSyntaxTree]]:
        """
        验证数据项
        :param item: 数据项，SpiderItem 对象
        :param section: 值为 "train" 或者 "val"
        :return:
        """
        # 如果本数据项所属数据库号 不在字典 self.sql_schemas 中，则添加之
        if item.spider_schema.db_id not in self.sql_schemas:
            self.sql_schemas[item.spider_schema.db_id] = preprocess_schema_uncached(
                schema=item.spider_schema,
                db_path=item.db_path,
                tokenize=self._schema_tokenize,
            )
        # print(item.question)
        # print(item.query)
        # print(item.spider_sql)
        try:
            # 检查合法性：item 是 SpiderItem 对象，且 self.transition_system 是 SpiderTransitionSystem 对象
            if isinstance(item, SpiderItem) and isinstance(
                self.transition_system, SpiderTransitionSystem
            ):
                # 构建 AST 树，返回 asdl_ast 为 AbstractSyntaxTree 抽象语法树对象
                # item.spider_sql 是源数据中的 "sql" 字段 (解析成了 dict)
                asdl_ast = self.transition_system.surface_code_to_ast(
                    code=item.spider_sql
                )
            else:
                raise NotImplementedError
            return True, asdl_ast
        except Exception as e:
            if "train" not in section:
                # TODO (YuweiYin) 疑问：代码 return True, None 不会被执行，如果 "train" 不在 section 里，该报错还是正常返回？
                raise e
                return True, None
            else:
                raise e

    def preprocess_item(
        self,
        item: SpiderItem,
        sql_schema: SQLSchema,
        validation_info: AbstractSyntaxTree,
    ) -> RATPreprocItem:
        raise NotImplementedError

    def add_item(
        self, item: SpiderItem, section: str, validation_info: AbstractSyntaxTree
    ) -> None:
        """Adds item and copies of it with shuffled schema if num_schema_shuffles > 0
        对数据项 item 进行预处理，并加入到词表中
        """
        # 获取该数据项所属的数据库名
        sql_schema = self.sql_schemas[item.spider_schema.db_id]
        # 根据数据库名 sql_schema 和抽象语法树结构 validation_info，返回本数据项 item 经处理后的 RATPreprocItem 对象
        preproc_item_no_shuffle = self.preprocess_item(
            item, sql_schema, validation_info
        )
        
        preproc_items = [preproc_item_no_shuffle]

        # 根据配置，决定要随机打乱多少数据库
        # 默认配置中 train_num_schema_shuffles: 0, val_num_schema_shuffles: 0
        if "train" in section:
            num_schema_shuffles = self.train_num_schema_shuffles
        elif "val" in section:
            num_schema_shuffles = self.val_num_schema_shuffles
        else:
            num_schema_shuffles = 0
        for _ in range(num_schema_shuffles):
            shuffled_schema = shuffle_schema(sql_schema)
            preproc_items.append(
                replace(preproc_item_no_shuffle, sql_schema=shuffled_schema)
            )

        # 分别添加 train 和 val 的(处理后的)数据项
        if section not in self.preproc_items:
            self.preproc_items[section] = []
        self.preproc_items[section] += preproc_items

        # 如果有 train 数据，则更新词表
        if "train" in section:
            self.update_vocab(item, preproc_item_no_shuffle)

    def clear_items(self) -> None:
        self.preproc_items: Dict[str, List[RATPreprocItem]] = {}

    def update_vocab(self, item: SpiderItem, preproc_item: RATPreprocItem):
        raise NotImplementedError

    def save_examples(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        for section, items in self.preproc_items.items():
            with open(os.path.join(self.save_path, section + ".pkl"), "wb") as f:
                pickle.dump(items, f)

    def save(self) -> None:
        self.save_examples()

        # production rules + Reduce + MASK + GenToken tokens that are *not* in the encoder sequence
        for element in itertools.chain(
            map(
                lambda production: ApplyRuleAction(production=production),
                self.transition_system.grammar.id2prod.values(),
            ),
            (ReduceAction(), MaskAction()),
        ):
            self.target_vocab_counter[element] = self.min_freq

        # 存储字典
        self.target_vocab = ActionVocab(
            counter=self.target_vocab_counter,
            max_size=50000,
            min_freq=self.min_freq,
            specials=[ActionVocab.UNK],
        )
        with open(self.target_vocab_path, "wb") as f:
            pickle.dump(self.target_vocab, f)

    def load(self) -> None:
        with open(self.target_vocab_path, "rb") as f:
            self.target_vocab = pickle.load(f)

    def dataset(self, section: str) -> List[RATPreprocItem]:
        with open(os.path.join(self.save_path, section + ".pkl"), "rb") as f:
            items = pickle.load(f)
        return items


class SingletonGloVe(Vectors):

    _glove = None

    def __init__(self):
        SingletonGloVe._load_if_needed()

    @property
    def vectors(self):
        return SingletonGloVe._glove.vectors

    def __getitem__(self, token):
        return SingletonGloVe._glove[token]

    @property
    def dim(self):
        return self._glove.dim

    @property
    def stoi(self):
        return SingletonGloVe._glove.stoi

    @staticmethod
    def _load_if_needed():
        if SingletonGloVe._glove is None:
            SingletonGloVe._glove = GloVe(
                name="42B",
                dim=300,
                cache=os.path.join(
                    os.environ.get("CACHE_DIR", os.getcwd()), ".vector_cache"
                ),
            )

    def __setstate__(self, state):
        assert len(state) == 0
        self._load_if_needed()


@registry.register("preproc", "TransformerDuoRAT")
class TransformerDuoRATPreproc(DuoRATPreproc):
    def __init__(self, **kwargs) -> None:
        super(TransformerDuoRATPreproc, self).__init__(**kwargs)

        self.use_full_glove_vocab = kwargs.get("use_full_glove_vocab", False)

        # for GloVe tokens that appear in the training data
        self.input_vocab_a_counter = Counter()
        self.input_vocab_a_vectors = SingletonGloVe()
        self.input_vocab_a_path = os.path.join(self.save_path, "input_vocab_a.pkl")
        self.input_vocab_a = None

        # for tokens that appear in the training data and are not in GloVe
        self.input_vocab_b_counter = Counter()
        self.input_vocab_b_path = os.path.join(self.save_path, "input_vocab_b.pkl")
        self.input_vocab_b = None

        # self.grouped_payload: Dict[str, List[dict]] = defaultdict(list)

    def input_a_str_to_id(self, s: str) -> int:
        return self.input_vocab_a.__getitem__(s)

    def input_b_str_to_id(self, s: str) -> int:
        return self.input_vocab_b.__getitem__(s)

    def _schema_tokenize(
        self, type: Optional[str], something: List[str], name: str
    ) -> List[str]:
        return (
            ["<type: {}>".format(type)] if type is not None else []
        ) + self.tokenizer.tokenize(name)

    def preprocess_item(
        self,
        item: SpiderItem,
        sql_schema: SQLSchema,
        validation_info: AbstractSyntaxTree,
    ) -> RATPreprocItem:
        slml_question: str = self.schema_linker.question_to_slml(
            question=item.question, sql_schema=sql_schema,
        ) if item.slml_question is None else item.slml_question

        item.slml_question = slml_question
        # self.grouped_payload[sql_schema.db_id].append(item.orig)

        parser = SLMLParser(sql_schema=sql_schema, tokenizer=self.tokenizer)
        parser.feed(data=slml_question)
        parser.close()

        question: Tuple[PreprocQuestionToken, ...] = parser.question_tokens

        asdl_ast = validation_info
        actions = (
            tuple(self.transition_system.get_actions(asdl_ast))
            if asdl_ast is not None
            else tuple()
        )
        return RATPreprocItem(question=question, sql_schema=sql_schema, actions=actions)

    def update_vocab(self, item: SpiderItem, preproc_item: RATPreprocItem):
        if item.spider_schema.db_id in self.counted_db_ids:
            # tokens_to_count: List[str] = [token.value for token in question]
            tokens_to_count: List[str] = self.tokenizer.tokenize(item.question)
        else:
            self.counted_db_ids.add(item.spider_schema.db_id)
            tokens_to_count: List[str] = list(
                itertools.chain(
                    # (token.value for token in question),
                    self.tokenizer.tokenize(item.question),
                    *preproc_item.sql_schema.tokenized_column_names.values(),
                    *preproc_item.sql_schema.tokenized_table_names.values()
                )
            )

        # add to first input vocab only what is in GLoVe
        self.input_vocab_a_counter.update(
            (
                token
                for token in tokens_to_count
                if token in self.input_vocab_a_vectors.stoi
            )
        )

        # add only to second input vocab what is *not* already in first input vocab (GLoVe)
        self.input_vocab_b_counter.update(
            (
                token
                for token in tokens_to_count
                if token not in self.input_vocab_a_vectors.stoi
            )
        )

        # add only GenToken tokens to target vocab that are *not* in the encoder sequence
        self.target_vocab_counter.update(
            (
                action
                for action in all_spider_gen_token_actions(preproc_item.actions)
                if action.token not in tokens_to_count
            )
        )

    def save(self) -> None:
        super(TransformerDuoRATPreproc, self).save()

        # GloVe tokens that appear in the training data
        self.input_vocab_a = Vocab(
            counter=self.input_vocab_a_counter,
            max_size=50000,
            min_freq=1,
            vectors=self.input_vocab_a_vectors,
            specials=["<unk>"],
            specials_first=True,
        )
        with open(self.input_vocab_a_path, "wb") as f:
            pickle.dump(self.input_vocab_a, f)

        # tokens that appear in the training data and are not in GloVe
        self.input_vocab_b = Vocab(
            counter=self.input_vocab_b_counter,
            max_size=5000,
            min_freq=self.min_freq,
            specials=["<unk>"],
            specials_first=True,
        )
        with open(self.input_vocab_b_path, "wb") as f:
            pickle.dump(self.input_vocab_b, f)

    def load(self) -> None:
        super(TransformerDuoRATPreproc, self).load()

        if self.use_full_glove_vocab:
            glove_with_fake_freqs = {
                token: len(self.input_vocab_a_vectors) - index
                for token, index in self.input_vocab_a_vectors.stoi.items()
            }
            # by setting ',' frequency to 0 we make it an unknown word
            # this is ATM necessary for backward compatibility
            glove_with_fake_freqs[","] = 0
            self.input_vocab_a = Vocab(
                Counter(glove_with_fake_freqs), specials=["<unk>"], specials_first=True,
            )
        else:
            with open(self.input_vocab_a_path, "rb") as f:
                self.input_vocab_a = pickle.load(f)
        with open(self.input_vocab_b_path, "rb") as f:
            self.input_vocab_b = pickle.load(f)


@registry.register("preproc", "BertDuoRAT")
class BertDuoRATPreproc(DuoRATPreproc):
    tokenizer: BERTTokenizer

    def __init__(self, **kwargs) -> None:
        super(BertDuoRATPreproc, self).__init__(**kwargs)
        # 默认配置中 add_cls_token: true, add_sep_token: false,
        self.add_cls_token = kwargs["add_cls_token"]
        self.add_sep_token = kwargs["add_sep_token"]
        assert isinstance(self.tokenizer, BERTTokenizer)

    def input_a_str_to_id(self, s: str) -> int:
        return self.tokenizer.convert_token_to_id(s)

    def input_b_str_to_id(self, s: str) -> int:
        return 0

    def _schema_tokenize(
        self, type: Optional[str], something: List[str], name: str
    ) -> List[str]:
        return (
            ([self.tokenizer.cls_token] if self.add_cls_token else [])
            + self.tokenizer.tokenize(
                ("{} ".format(type) if type is not None else "") + name
            )
            + ([self.tokenizer.sep_token] if self.add_sep_token else [])
        )

    def preprocess_item(
        self,
        item: SpiderItem,
        sql_schema: SQLSchema,
        validation_info: AbstractSyntaxTree,
    ) -> RATPreprocItem:
        """预处理数据项"""

        slml_question: str = self.schema_linker.question_to_slml(
            question=item.question, sql_schema=sql_schema,
        ) if item.slml_question is None else item.slml_question
        item.slml_question = slml_question

        # print(item.question)
        # print(slml_question)
        # print('-------------------')
        # print(self.tokenizer)

        # SLML: Schema Linking Markup Language
        # 不同的数据库 sql_schema 会有不同的 SLMLParser 对象
        parser = SLMLParser(sql_schema=sql_schema, tokenizer=self.tokenizer)
        parser.feed(data=slml_question)
        parser.close()

        # 拼接原始 question 和语法解析后的 question
        # 默认配置中 add_cls_token: true, add_sep_token: false,
        question: Tuple[PreprocQuestionToken, ...] = (
            (
                PreprocQuestionToken(
                    key=QuestionTokenId(uuid4()), value=self.tokenizer.cls_token
                ),
            )
            if self.add_cls_token
            else tuple()
        ) + parser.question_tokens + (
            (
                PreprocQuestionToken(
                    key=QuestionTokenId(uuid4()), value=self.tokenizer.sep_token
                ),
            )
            if self.add_sep_token
            else tuple()
        )

        # for x in question:
        #     print(x)

        # flag = 0
        # for x in question: 
        #     if len(x.match_tags)>0:
        #         if flag == 0:
        #             print('[', end='')  
        #             flag = 1 
        #     if len(x.match_tags)==0:
        #         if flag == 1:
        #             print(']',end='')
        #             flag = 0
        #     print(x.value, end='')
            
        # print('\n')
        # print('------------------------')

        # 输入的 AbstractSyntaxTree 对象 validation_info 即为 asdl_ast
        asdl_ast = validation_info
        
        # print(asdl_ast)
        # print('---------------------------')

        # 根据当前抽象语法树对象 asdl_ast 获得 actions 动作 TODO (YuweiYin) self.transition_system.get_actions 函数分析
        actions = tuple(self.transition_system.get_actions(asdl_ast))
        # 返回 RAT 预处理后的数据项。RATPreprocItem 是一个数据类 @dataclass(order=True, frozen=True)，含三个字段
        return RATPreprocItem(question=question, sql_schema=sql_schema, actions=actions)

    def update_vocab(self, item: SpiderItem, preproc_item: RATPreprocItem):
        """更新词表"""
        if item.spider_schema.db_id in self.counted_db_ids:
            tokens_to_count: List[str] = [
                token.value for token in preproc_item.question
            ]
        else:
            self.counted_db_ids.add(item.spider_schema.db_id)
            tokens_to_count: List[str] = list(
                itertools.chain(
                    (token.value for token in preproc_item.question),
                    *preproc_item.sql_schema.tokenized_column_names.values(),
                    *preproc_item.sql_schema.tokenized_table_names.values()
                )
            )

        # add only GenToken tokens to target vocab that are *not* in the encoder sequence
        # for action in all_spider_gen_token_actions(preproc_item.actions):
        temp1 =  [action for action in all_spider_gen_token_actions(preproc_item.actions)]
        temp2 =  [action.token for action in all_spider_gen_token_actions(preproc_item.actions)]

        self.target_vocab_counter.update(
            (
                action
                for action in all_spider_gen_token_actions(preproc_item.actions)
                if action.token not in tokens_to_count
            )
        )

    def save(self) -> None:
        super(BertDuoRATPreproc, self).save()

    def load(self) -> None:
        super(BertDuoRATPreproc, self).load()
