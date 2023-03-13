import os
import sys

import corenlp
import requests


class CoreNLP:
    def __init__(self):
        # 若无 CORENLP_HOME 环境变量，则设置之
        if not os.environ.get("CORENLP_HOME"):
            os.environ["CORENLP_HOME"] = os.path.abspath(
                # 此处的路径根据当前项目 stanford-corenlp 文件的实际位置来设置
                os.path.join(
                    '../../stanford-corenlp-4.1.0'
                )
            )
        # 如果 CORENLP_HOME 环境变量所指的文件不存在，则报错
        if not os.path.exists(os.environ["CORENLP_HOME"]):
            raise Exception(
                """Please install Stanford CoreNLP and put it at {}.

                Direct URL: http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
                Landing page: https://stanfordnlp.github.io/CoreNLP/""".format(
                    os.environ["CORENLP_HOME"]
                )
            )
        self.client = corenlp.CoreNLPClient()

    def __del__(self):
        self.client.stop()

    def annotate(self, text, annotators=None, output_format=None, properties=None):
        # 尝试进行 annotate 操作，若失败则重启并重试
        try:
            result = self.client.annotate(text, annotators, properties)
        except (
            corenlp.client.PermanentlyFailedException,
            requests.exceptions.ConnectionError,
        ) as e:
            print(
                "\nWARNING: CoreNLP connection timeout. Recreating the server...",
                file=sys.stderr,
            )
            # 重启 CoreNLPClient
            self.client.stop()
            self.client.start()
            # 重试 annotate 操作
            result = self.client.annotate(text, annotators,properties)

        return result


# (单例) CoreNLP 全局对象
_singleton = None


def annotate(text, annotators=None, properties=None):
    # 声明使用 _singleton 全局对象
    global _singleton

    # 若无 CoreNLP 对象，则创建之
    if not _singleton:
        _singleton = CoreNLP()

    # 返回 CoreNLP 处理后的结果
    return _singleton.annotate(text, annotators, properties)
