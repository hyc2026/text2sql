import abc
from functools import lru_cache
from typing import List, Sequence, Tuple


import stanza
import jieba 

from transformers import BertTokenizerFast
from transformers import AutoTokenizer
from transformers import DistilBertTokenizerFast
from transformers import ElectraTokenizerFast
from duorat.utils import registry, corenlp


class AbstractTokenizer(metaclass=abc.ABCMeta):
    """分词器抽象基类，主要含 编码 和 解码 两个函数"""

    @abc.abstractmethod
    def tokenize(self, s: str) -> List[str]:
        pass

    @abc.abstractmethod
    def detokenize(self, xs: Sequence[str]) -> str:
        pass


@registry.register("tokenizer", "JiebaTokenizer")
class JiebaTokenizer(AbstractTokenizer):
    def __init__(self):
        pass

    def _tokenize(self, s: str) -> List[str]:
        ann = jieba.cut(s) 
        return ann 

    def tokenize(self, s: str) -> List[str]:
        return [token.lower() for token in self._tokenize(s)]

    def tokenize_with_raw(self, s: str) -> List[Tuple[str, str]]:
        return [(token.lower(), token) for token in self._tokenize(s)]

    def detokenize(self, xs: Sequence[str]) -> str:
        return " ".join(xs)    


# TODO (LinzhengChai)
# class LacTokenizer(AbstractTokenizer):


@registry.register("tokenizer", "CoreNLPTokenizer")
class CoreNLPTokenizer(AbstractTokenizer):
    def __init__(self):
        pass

    @lru_cache(maxsize=1024)
    def _tokenize(self, s: str) -> List[str]:
        # 使用 Stanford CoreNLP 工具进行分词 token 化
        ann = corenlp.annotate(
            text=s,
            annotators=["tokenize", "ssplit"],
            properties={
                "outputFormat": "serialized",
                "tokenize.options": "asciiQuotes = false, latexQuotes=false, unicodeQuotes=false, ",
            },
        )
        # 返回所有 ann.sentence[i].token[j].word 组成的列表
        return [tok.word for sent in ann.sentence for tok in sent.token]

    def tokenize(self, s: str) -> List[str]:
        # 先用 Stanford CoreNLP 工具进行分词 token 化，然后转小写
        return [token.lower() for token in self._tokenize(s)]

    def tokenize_with_raw(self, s: str) -> List[Tuple[str, str]]:
        # 返回二元组的列表，每个二元组包含处理后 token 的小写形式 及其 原始形式
        return [(token.lower(), token) for token in self._tokenize(s)]

    def detokenize(self, xs: Sequence[str]) -> str:
        # 仅是简单地用单个空格把 str 列表连接成单个字符串
        return " ".join(xs)


@registry.register("tokenizer", "StanzaTokenizer")
class StanzaTokenizer(AbstractTokenizer):
    def __init__(self):
        # 下载 stanza 的英文 en 分词器，实例化分词器为 self.nlp
        stanza.download("en", processors="tokenize")
        self.nlp = stanza.Pipeline(lang="en", processors="tokenize")

    @lru_cache(maxsize=1024)
    def tokenize(self, s: str) -> List[str]:
        # 使用 stanza 英文分词器进行处理，返回所有 doc.sentences[i].tokens[j].question 组成的列表
        doc = self.nlp(s)
        return [
            token.question for sentence in doc.sentences for token in sentence.tokens
        ]

    def detokenize(self, xs: Sequence[str]) -> str:
        return " ".join(xs)


@registry.register("tokenizer", "BERTTokenizer")
class BERTTokenizer(AbstractTokenizer):
    def __init__(self, pretrained_model_name_or_path: str):
        # self._bert_tokenizer = BertTokenizerFast.from_pretrained(
        #     pretrained_model_name_or_path=pretrained_model_name_or_path
        # ) 
        # self._bert_tokenizer =  AutoTokenizer.from_pretrained('google/electra-small-discriminator')
        # self._bert_tokenizer =  ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
        self._bert_tokenizer =  ElectraTokenizerFast.from_pretrained(pretrained_model_name_or_path)

    def tokenize(self, s: str) -> List[str]:
        return self._bert_tokenizer.tokenize(s)

    def tokenize_with_raw(self, s: str) -> List[Tuple[str, str]]:
        # TODO: at some point, hopefully, transformers API will be mature enough
        # to do this in 1 call instead of 2
        tokens = self._bert_tokenizer.tokenize(s)
        # print("s",s)
        encoding_result = self._bert_tokenizer(s, return_offsets_mapping=True)
        # print('encoding result:', encoding_result)
        # print(self._bert_tokenizer(s))
        
        assert len(encoding_result[0]) == len(tokens) + 2
        raw_token_strings = [
            s[start:end] for start, end in encoding_result["offset_mapping"][1:-1]
        ]
        raw_token_strings_with_sharps = []
        for token, raw_token in zip(tokens, raw_token_strings):
            if not (token == raw_token.lower()
                    or token[2:] == raw_token.lower()
                    or token[-2:] == raw_token.lower()):
                print(token, raw_token)
            # assert (
            #     token == raw_token.lower()
            #     or token[2:] == raw_token.lower()
            #     or token[-2:] == raw_token.lower()
            # )
            if token.startswith("##"):
                raw_token_strings_with_sharps.append("##" + raw_token)
            elif token.endswith("##"):
                raw_token_strings_with_sharps.append(raw_token + "##")
            else:
                raw_token_strings_with_sharps.append(raw_token)
        return zip(tokens, raw_token_strings_with_sharps)

    def detokenize(self, xs: Sequence[str]) -> str:
        """Naive implementation, see https://github.com/huggingface/transformers/issues/36"""
        text = " ".join([x for x in xs])
        fine_text = text.replace(" ##", "")
        return fine_text

    def convert_token_to_id(self, s: str) -> int:
        return self._bert_tokenizer.convert_tokens_to_ids(s)

    @property
    def cls_token(self) -> str:
        return self._bert_tokenizer.cls_token

    @property
    def sep_token(self) -> str:
        return self._bert_tokenizer.sep_token
