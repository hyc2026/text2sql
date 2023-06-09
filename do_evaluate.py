
"""评价工具：计算预估 SQL 的精确匹配 ACCURACY。对于select、where等部分的
多个成分，如果仅是顺序不同也会算正确。
本脚本参考了 spider 数据集中公开的 evaluation.py 代码(https://github.com/taoyds/spider)。
"""

from __future__ import division

import copy
import json
import logging
import re
from collections import defaultdict
from io import open
from utils import clean_sql_query
################################
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id)
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, cond_op, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': [(agg_id, val_unit), (agg_id, val_unit), ...]
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [(agg_id, val_unit), ...])
#   'having': condition
#   'limit': None/number(int)
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

COND_OPS = ('not_in', 'between', '==', '>', '<', '>=', '<=', '!=', 'in', 'like')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

LOGIC_AND_OR = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

CONST_COLUMN = set(['time_now'])

EXPECT_BRACKET_PRE_TOKENS = set(AGG_OPS + SQL_OPS + COND_OPS + CLAUSE_KEYWORDS + ('from', ','))

HARDNESS = {
    "component1": ("where", "group", "order", "limit", "join", "or", "like"),
    "component2": ("except", "union", "intersect"),
}

LEVELS = ["easy", "medium", "hard", "extra", "all"]
PARTIAL_TYPES = [
    "select",
    "select(no AGG)",
    "where",
    "where(no OP)",
    "group(no Having)",
    "group",
    "order",
    "and/or",
    "IUEN",
    "keywords",
]

g_empty_sql = {"select": [], "from": {"conds": [], "table_units": []},
               "where": [], "groupBy": [], "having": [], "orderBy": [], "limit": None,
               "except": None, "intersect": None, "union": None}


#################################
def tokenize(string):
    """
    Args:

    Returns:
    """
    string = string.replace("\'", "\"").lower()
    assert string.count('"') % 2 == 0, "Unexpected quote"

    def _extract_value(string):
        """extract values in sql"""
        fields = string.split('"')
        for idx, tok in enumerate(fields):
            if idx % 2 == 1:
                fields[idx] = '"%s"' % (tok)
        return fields

    def _resplit(tmp_tokens, fn_split, fn_omit):
        """resplit"""
        new_tokens = []
        for token in tmp_tokens:
            token = token.strip()
            if fn_omit(token):
                new_tokens.append(token)
            elif re.match(r'\d\d\d\d-\d\d(-\d\d)?', token):
                new_tokens.append('"%s"' % (token))
            else:
                new_tokens.extend(fn_split(token))
        return new_tokens

    tokens_tmp = _extract_value(string)

    two_bytes_op = ['==', '!=', '>=', '<=', '<>', '<in>']
    sep1 = re.compile(r'([ \+\-\*/\(\),><;])')  # 单字节运算符
    sep2 = re.compile('(' + '|'.join(two_bytes_op) + ')')  # 多字节运算符
    tokens_tmp = _resplit(tokens_tmp, lambda x: x.split(' '), lambda x: x.startswith('"'))
    tokens_tmp = _resplit(tokens_tmp, lambda x: re.split(sep2, x), lambda x: x.startswith('"'))
    tokens_tmp = _resplit(tokens_tmp, lambda x: re.split(sep1, x),
                          lambda x: x in two_bytes_op or x.startswith('"'))
    tokens = list(filter(lambda x: x.strip() != '', tokens_tmp))

    def _post_merge(tokens):
        """merge:
              * col name with "(", ")"
              * values with +/-
        """
        idx = 1
        while idx < len(tokens):
            if tokens[idx] == '(' and tokens[idx - 1] not in EXPECT_BRACKET_PRE_TOKENS:
                while idx < len(tokens):
                    tmp_tok = tokens.pop(idx)
                    tokens[idx - 1] += tmp_tok
                    if tmp_tok == ')':
                        break
            elif tokens[idx] in ('+', '-') and tokens[idx - 1] in COND_OPS and idx + 1 < len(tokens):
                tokens[idx] += tokens[idx + 1]
                tokens.pop(idx + 1)
                idx += 1
            else:
                idx += 1
        return tokens

    tokens = _post_merge(tokens)
    return tokens


def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        alias[toks[idx + 1]] = toks[idx - 1]
    return alias


def get_tables_with_alias(schema, toks):
    """
    Args:

    Returns:
    """
    tables = scan_alias(toks)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
    tok = toks[start_idx]
    if tok == "*":
        return start_idx + 1, schema.id_map[tok]
    if tok in CONST_COLUMN:
        return start_idx + 1, tok

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        return start_idx + 1, schema.id_map[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx + 1, schema.id_map[key]

    raise RuntimeError("Error col: {}".format(tok))


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id)

    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
    Args:

    Returns:
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    if unit_op in (UNIT_OPS.index('+'), UNIT_OPS.index('*')):
        col_unit1, col_unit2 = sorted([col_unit1, col_unit2])

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx + 1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.id_map[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
    Args:

    Returns:
    """
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    def _force_float(str_num):
        """force float, just for debug"""
        last = ''
        while len(str_num) > 0:
            try:
                n = float(str_num)
                if last == '%':
                    n /= 100
                return n
            except:
                last = str_num[-1]
                str_num = str_num[:-1]
        raise ValueError('not a float number')

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif toks[idx].startswith('"') and toks[idx].endswith('"'):  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val_str = toks[idx]
            # val = float(val_str) if val_str[-1] != '%' else float(val_str[:-1]) / 100
            val = _force_float(val_str)
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')' \
                    and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS \
                    and toks[end_idx] not in JOIN_KEYWORDS:
                end_idx += 1

            idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
            idx = end_idx

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
    Args:

    Returns:
    """
    idx = start_idx
    len_ = len(toks)
    conds = []

    while idx < len_:
        agg_id = 0
        if idx < len_ and toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1

        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)

        op_str = toks[idx]
        if op_str == 'not':
            assert toks[idx + 1] == 'in', '"not" must followed by "in"'
            op_str = 'not_in'
            idx += 1
        assert idx < len_ and op_str in COND_OPS, "Error condition: idx: {}, tok: {}".format(idx, op_str)
        op_id = COND_OPS.index(op_str)
        idx += 1
        val1 = val2 = None
        idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
        val2 = None

        conds.append((agg_id, op_id, val_unit, val1, val2))

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

        if idx < len_ and toks[idx] in LOGIC_AND_OR:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
    Args:

    Returns:
    """
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, val_units


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []
    last_table = None

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
            last_table = sql['from']['table_units'][0][1].strip('_')
        else:
            if idx < len_ and toks[idx] == 'join':
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'], table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        if idx < len_ and toks[idx] == 'a':
            assert last_table is not None, 'last_table should be a table name strin, not None'
            tables_with_alias['a'] = last_table
            idx += 2
        elif idx < len_ and toks[idx] == 'b':
            assert last_table is not None, 'last_table should be a table name strin, not None'
            tables_with_alias['b'] = last_table
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return [idx, table_units, conds, default_tables]


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    """
    Args:

    Returns:
    """
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    """
    Args:

    Returns:
    """
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    """
    Args:

    Returns:
    """
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc'  # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    """
    Args:

    Returns:
    """
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    """
    Args:

    Returns:
    """
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx - 1])

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema):
    """
    Args:

    Returns:
    """
    isBlock = False  # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(toks, idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['orderBy'] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def load_data(fpath):
    """
    Args:

    Returns:
    """
    with open(fpath, encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_sql(schema, query):
    """
    Args:

    Returns:
    """
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)

    return sql


def skip_semicolon(toks, start_idx):
    """
    Args:

    Returns:
    """
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx


#################################

def update_scores_match(scores, exact_score, hardness, partial_scores, partial_types):
    scores[hardness]["exact"] += exact_score
    scores["all"]["exact"] += exact_score
    for type_ in partial_types:
        if partial_scores[type_]["pred_total"] > 0:
            scores[hardness]["partial"][type_]["acc"] += partial_scores[
                type_
            ]["acc"]
            scores[hardness]["partial"][type_]["acc_count"] += 1
        if partial_scores[type_]["label_total"] > 0:
            scores[hardness]["partial"][type_]["rec"] += partial_scores[
                type_
            ]["rec"]
            scores[hardness]["partial"][type_]["rec_count"] += 1
        scores[hardness]["partial"][type_]["f1"] += partial_scores[type_][
            "f1"
        ]
        if partial_scores[type_]["pred_total"] > 0:
            scores["all"]["partial"][type_]["acc"] += partial_scores[
                type_
            ]["acc"]
            scores["all"]["partial"][type_]["acc_count"] += 1
        if partial_scores[type_]["label_total"] > 0:
            scores["all"]["partial"][type_]["rec"] += partial_scores[
                type_
            ]["rec"]
            scores["all"]["partial"][type_]["rec_count"] += 1
        scores["all"]["partial"][type_]["f1"] += partial_scores[type_][
            "f1"
        ]


class Evaluator(object):
    """A simple evaluator"""

    def __init__(self):
        """init"""
        self.partial_scores = None

        self.scores = {
            level: {
                "count": 0,
                "partial": {
                    type_: {
                        "acc": 0.0,
                        "rec": 0.0,
                        "f1": 0.0,
                        "acc_count": 0,
                        "rec_count": 0,
                    }
                    for type_ in PARTIAL_TYPES
                },
                "exact": 0.0,
                "exec": 0,
            }
            for level in LEVELS
        }

    def _eval_exact_match(self, pred, gold):
        """eval_exact_match"""
        partial_scores, _, _, _ = self.eval_partial_match(pred, gold)
        self.partial_scores = partial_scores

        for _, score in partial_scores.items():
            # 只有出现不完全匹配的情况就返回0
            if score['f1'] != 1:
                return 0
        if len(gold['from']['table_units']) > 0:
            # 判断 from 选择的表是否完全一致（统一重排序后）
            gold_tables = sorted(gold['from']['table_units'], key=lambda x: str(x))
            pred_tables = sorted(pred['from']['table_units'], key=lambda x: str(x))
            return gold_tables == pred_tables
        return 1

    def eval_exact_match(self, pred, gold):
        """wrapper of evaluate examct match, to process
        `SQL1 intersect/union SQL2` vs `SQL2 intersect/union SQL1`
        a   union   b  <=>  b   union   a
        a intersect b  <=>  b intersect a

        Args:
            pred (TYPE): 预测值 结构化json
            gold (TYPE): 真实值 结构化json

        Returns: 1 --- 如果完全match（包括交换后match的情况）
                 0 --- 其他情况

        Raises: NULL
        """
        # 直接计算原始是否 exact match
        score = self._eval_exact_match(pred, gold)
        if score == 1:
            return score

        # 如果存在union, 则交换gold的union部分再计算
        if gold['union'] is not None:
            new_gold = gold['union']
            gold['union'] = None
            new_gold['union'] = gold
            return self._eval_exact_match(pred, new_gold)
        # 同理，如果存在 intersect, 则交换gold的 intersect部分再计算
        elif gold['intersect'] is not None:
            new_gold = gold['intersect']
            gold['intersect'] = None
            new_gold['intersect'] = gold
            return self._eval_exact_match(pred, new_gold)
        else:
            return 0

    def eval_partial_match(self, pred, gold):
        """eval_partial_match"""
        res = {}

        sum_gold_total = 0  # 总label子组件个数
        sum_pred_total = 0  # 总pred 子组件个数
        sum_cnt = 0  # 匹配上的子组件个数

        gold_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['select'] = {'acc': acc, 'rec': rec, 'f1': f1, 'gold_total': gold_total, 'pred_total': pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, gold_total)
        res['select(no AGG)'] = {'acc': acc, 'rec': rec, 'f1': f1, 'gold_total': gold_total, 'pred_total': pred_total}
        sum_gold_total += gold_total
        sum_pred_total += pred_total
        sum_cnt += cnt

        gold_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['where'] = {'acc': acc, 'rec': rec, 'f1': f1, 'gold_total': gold_total, 'pred_total': pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, gold_total)
        res['where(no OP)'] = {'acc': acc, 'rec': rec, 'f1': f1, 'gold_total': gold_total, 'pred_total': pred_total}
        sum_gold_total += gold_total
        sum_pred_total += pred_total
        sum_cnt += cnt

        gold_total, pred_total, cnt = eval_group(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['group(no Having)'] = {'acc': acc, 'rec': rec, 'f1': f1, 'gold_total': gold_total, 'pred_total': pred_total}
        sum_gold_total += gold_total
        sum_pred_total += pred_total
        sum_cnt += cnt

        gold_total, pred_total, cnt = eval_having(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['group'] = {'acc': acc, 'rec': rec, 'f1': f1, 'gold_total': gold_total, 'pred_total': pred_total}
        sum_gold_total += gold_total
        sum_pred_total += pred_total
        sum_cnt += cnt

        gold_total, pred_total, cnt = eval_order(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['order'] = {'acc': acc, 'rec': rec, 'f1': f1, 'gold_total': gold_total, 'pred_total': pred_total}
        sum_gold_total += gold_total
        sum_pred_total += pred_total
        sum_cnt += cnt

        gold_total, pred_total, cnt = eval_and_or(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['and/or'] = {'acc': acc, 'rec': rec, 'f1': f1, 'gold_total': gold_total, 'pred_total': pred_total}
        sum_gold_total += gold_total
        sum_pred_total += pred_total
        sum_cnt += cnt

        gold_total, pred_total, cnt = eval_IUEN(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['IUEN'] = {'acc': acc, 'rec': rec, 'f1': f1, 'gold_total': gold_total, 'pred_total': pred_total}
        sum_gold_total += gold_total
        sum_pred_total += pred_total
        sum_cnt += cnt

        gold_total, pred_total, cnt = eval_keywords(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['keywords'] = {'acc': acc, 'rec': rec, 'f1': f1, 'gold_total': gold_total, 'pred_total': pred_total}

        return res, sum_gold_total, sum_pred_total, sum_cnt

    def evaluate_one(self, db_name, gold, predicted):
        schema = self.schemas[db_name]
        g_sql = get_sql(schema, gold)
        # self.scores["all"]["count"] += 1

        parse_error = False
        try:
            p_sql = get_sql(schema, predicted)
        except:
            # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            p_sql = {
                "except": None,
                "from": {"conds": [], "table_units": []},
                "groupBy": [],
                "having": [],
                "intersect": None,
                "limit": None,
                "orderBy": [],
                "select": [False, []],
                "union": None,
                "where": [],
            }

            # TODO fix
            parse_error = True

        # rebuild sql for value evaluation
        kmap = self.kmaps[db_name]
        g_valid_col_units = build_valid_col_units(g_sql["from"]["table_units"], schema)
        #       g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
        p_valid_col_units = build_valid_col_units(p_sql["from"]["table_units"], schema)
        #        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

        # if self.etype in ["all", "exec"]:
        #     self.scores[hardness]["exec"] += eval_exec_match(
        #         self.db_paths[db_name], predicted, gold, p_sql, g_sql
        #     )
        exact_score = None
        partial_scores = None
        if self.etype in ["all", "match"]:
            partial_scores, _, _, _, = self.eval_partial_match(p_sql, g_sql)
            exact_score = self.eval_exact_match(p_sql, g_sql, partial_scores)
            update_scores_match(self.scores, exact_score, 'easy', partial_scores, PARTIAL_TYPES)

        # return {
        #     "predicted": predicted,
        #     "gold": gold,
        #     "predicted_parse_error": parse_error,
        #     "hardness": hardness,
        #     "exact": exact_score,
        #     "partial": partial_scores,
        # }
        return {
            "predicted": predicted,
            "gold": gold,
            "predicted_parse_error": parse_error,
            "hardness": 'easy',
            "exact": exact_score,
            "partial": partial_scores,
        }


class Schema(object):
    """
    Simple schema which maps table&column to a unique identifier
    """

    def __init__(self, db):
        """init"""
        self._schema = self._build_schema(db)
        self._id_map = self._map(self._schema)

    @property
    def schema(self):
        """_schema property"""
        return self._schema

    @property
    def id_map(self):
        """_id_map property"""
        return self._id_map

    def _build_schema(self, db):
        """build <table, list of columns> schema by input db

        Args:
            db (dict): NULL

        Returns: TODO

        Raises: NULL
        """
        tables = [x.lower() for x in db['table_names']]
        dct_table2cols = defaultdict(list)
        for table_id, column in db['column_names']:
            if table_id < 0:
                continue
            dct_table2cols[tables[table_id]].append(column.lower())
        return dct_table2cols

    def _map(self, schema):
        """map"""
        id_map = {'*': "__all__"}
        for key, vals in schema.items():
            for val in vals:
                id_map[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"

        for key in schema:
            id_map[key.lower()] = "__" + key.lower() + "__"

        return id_map


def get_scores(count, pred_total, gold_total):
    """
    Args:

    Returns:
    """
    # 如果数量上都不一致，直接全0
    if pred_total != gold_total:
        return 0, 0, 0
    # 如果命中个数和标准答案个数一样（即全对），则全部返回1
    elif count == pred_total:
        return 1, 1, 1
    # TODO(wb) 如果是 pred_total == gold_total 且 0 < count < pred_total 这种情况怎么算分呢？
    return 0, 0, 0


def eval_sel(pred, gold):
    """
    Args:

    Returns:
    """
    pred_sel = copy.deepcopy(pred['select'])
    gold_sel = copy.deepcopy(gold['select'])
    # gold col_id without agg
    gold_wo_agg = [unit[1] for unit in gold_sel]
    # pred的unit个数
    pred_total = len(pred_sel)
    # gold的unit个数
    gold_total = len(gold_sel)
    # 命中个数
    cnt = 0
    # 去掉agg命中个数
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in gold_sel:
            cnt += 1
            gold_sel.remove(unit)
        if unit[1] in gold_wo_agg:
            cnt_wo_agg += 1
            gold_wo_agg.remove(unit[1])

    return [gold_total, pred_total, cnt, cnt_wo_agg]


def eval_where(pred, gold):
    """
    计算 where 组件的命中个数。
    对于 [cond_unit1, 'and/or', cond_unit2] 这种情况，计算过程直接忽略了 'and/or';
    对于 cnt_wo_agg，只计算 cond_unit 里 value_unit 的命中个数
    """
    # 这个意思就是在where这个组件的list里，每次跳2步（即跳过了and/or），留下的都是 cond_unit
    pred_conds = [unit for unit in pred['where'][::2]]
    gold_conds = [unit for unit in gold['where'][::2]]
    # 这里只取 cond_unit 里的 val_unit(calc_op, col_unit1, col_unit2)
    gold_wo_agg = [unit[2] for unit in gold_conds]
    pred_total = len(pred_conds)
    gold_total = len(gold_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in gold_conds:
            cnt += 1
            gold_conds.remove(unit)
        if unit[2] in gold_wo_agg:
            cnt_wo_agg += 1
            gold_wo_agg.remove(unit[2])

    return [gold_total, pred_total, cnt, cnt_wo_agg]


def eval_group(pred, gold):
    """
    计算 group by 组件的命中个数。
    """
    # 这个直接取每个 col_unit 的  col_id 了，丢掉了 agg_id
    pred_cols = [unit[1] for unit in pred['groupBy']]
    gold_cols = [unit[1] for unit in gold['groupBy']]
    pred_total = len(pred_cols)
    gold_total = len(gold_cols)
    cnt = 0
    # 如果有 T1.b 这种情况，就按.切分，取b，其实这样计算很粗略的
    pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
    gold_cols = [gold.split(".")[1] if "." in gold else gold for gold in gold_cols]
    for col in pred_cols:
        if col in gold_cols:
            cnt += 1
            gold_cols.remove(col)
    return [gold_total, pred_total, cnt]


def eval_having(pred, gold):
    """
    计算 having 组件的命中个数。至多只有1个组件。
    """
    pred_total = gold_total = cnt = 0
    # 前提是 groupBy 存在
    if len(pred['groupBy']) > 0:
        pred_total = 1
    if len(gold['groupBy']) > 0:
        gold_total = 1

    # 去掉了 col_unit 的 agg_id，留下 col_id
    pred_cols = [unit[1] for unit in pred['groupBy']]
    gold_cols = [unit[1] for unit in gold['groupBy']]

    # 如果 groupBy两者数量都为1，且col_id一致，且having内condition完全一致，则cnt=1，其他全为0
    # 经过统计分析，groupBy如果不为空，里面只有1列
    if pred_total == gold_total == 1 \
            and pred_cols == gold_cols \
            and pred['having'] == gold['having']:
        cnt = 1

    return [gold_total, pred_total, cnt]


def eval_order(pred, gold):
    """
    计算 order by 组件的命中个数。至多只有1个组件。
    """
    pred_total = gold_total = cnt = 0
    if len(pred['orderBy']) > 0:
        pred_total = 1
    if len(gold['orderBy']) > 0:
        gold_total = 1

    # 如果 oderby 和 limit 的数字完全相同，则 cnt = 1
    if len(gold['orderBy']) > 0 and pred['orderBy'] == gold['orderBy'] and pred['limit'] == gold['limit']:
        cnt = 1

    return [gold_total, pred_total, cnt]


def eval_and_or(pred, gold):
    """
    计算where组件中 and/or 的命中数。
    """
    # 从1开始，跳2步，取所有的 and/or
    pred_ao = pred['where'][1::2]
    gold_ao = gold['where'][1::2]
    # 其实这样也很粗糙
    pred_ao = set(pred_ao)
    gold_ao = set(gold_ao)

    if pred_ao == gold_ao:
        return 1, 1, 1
    return [len(pred_ao), len(gold_ao), 0]


def get_nestedSQL(sql):
    """
    Args:

    Returns:
    """
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    ##
    for from_nest_sql in [table_unit[1] for table_unit in sql['from']['table_units'] if table_unit[0] == 'sql']:
        nested.append(from_nest_sql)

    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def eval_nested(pred, gold):
    """
    Args:

    Returns:
    """
    gold_total = 0
    pred_total = 0
    cnt = 0
    if pred is not None:
        pred_total += 1
    if gold is not None:
        gold_total += 1
    if pred is not None and gold is not None:
        cnt += Evaluator().eval_exact_match(pred, gold)
    return [gold_total, pred_total, cnt]


def eval_IUEN(pred, gold):
    """
    Args:

    Returns:
    """
    lt1, pt1, cnt1 = eval_nested(pred['intersect'], gold['intersect'])
    lt2, pt2, cnt2 = eval_nested(pred['except'], gold['except'])
    lt3, pt3, cnt3 = eval_nested(pred['union'], gold['union'])
    gold_total = lt1 + lt2 + lt3
    pred_total = pt1 + pt2 + pt3
    cnt = cnt1 + cnt2 + cnt3
    return [gold_total, pred_total, cnt]


def get_keywords(sql):
    """
    Args:

    Returns:
    """
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')

    # or keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    ## TODO
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == COND_OPS.index('in')]) > 0:
        res.add('in')

    # like keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == COND_OPS.index('like')]) > 0:
        res.add('like')

    return res


def eval_keywords(pred, gold):
    """
    Args:

    Returns:
    """
    pred_keywords = get_keywords(pred)
    gold_keywords = get_keywords(gold)
    pred_total = len(pred_keywords)
    gold_total = len(gold_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in gold_keywords:
            cnt += 1
    return [gold_total, pred_total, cnt]


# Rebuild SQL functions for foreign key evaluation
def build_valid_col_units(table_units, schema):
    """
    Args:

    Returns:
    """
    col_ids = [table_unit[1] for table_unit in table_units if table_unit[0] == TABLE_TYPE['table_unit']]
    prefixs = [col_id[: -2] for col_id in col_ids]
    valid_col_units = []
    for value in schema.id_map.values():
        if '.' in value and value[:value.index('.')] in prefixs:
            valid_col_units.append(value)
    return valid_col_units


def rebuild_col_unit_col(valid_col_units, col_unit, kmap):
    """
    Args:

    Returns:
    """
    if col_unit is None:
        return col_unit

    agg_id, col_id = col_unit
    if col_id in kmap and col_id in valid_col_units:
        col_id = kmap[col_id]
    return agg_id, col_id


def rebuild_val_unit_col(valid_col_units, val_unit, kmap):
    """
    Args:

    Returns:
    """
    if val_unit is None:
        return val_unit

    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, kmap)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, kmap)
    return [unit_op, col_unit1, col_unit2]


def rebuild_table_unit_col(valid_col_units, table_unit, kmap):
    """
    Args:

    Returns:
    """
    if table_unit is None:
        return table_unit

    table_type, col_unit_or_sql = table_unit
    if isinstance(col_unit_or_sql, tuple):
        col_unit_or_sql = rebuild_col_unit_col(valid_col_units, col_unit_or_sql, kmap)
    return table_type, col_unit_or_sql


def rebuild_cond_unit_col(valid_col_units, cond_unit, kmap):
    """
    Args:

    Returns:
    """
    if cond_unit is None:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    val_unit = rebuild_val_unit_col(valid_col_units, val_unit, kmap)
    return [not_op, op_id, val_unit, val1, val2]


def rebuild_condition_col(valid_col_units, condition, kmap):
    """
    Args:

    Returns:
    """
    for idx in range(len(condition)):
        if idx % 2 == 0:
            condition[idx] = rebuild_cond_unit_col(valid_col_units, condition[idx], kmap)
    return condition


def rebuild_select_col(valid_col_units, sel, kmap):
    """
    Args:

    Returns:
    """
    if sel is None:
        return sel
    new_list = []
    for it in sel:
        agg_id, val_unit = it
        new_list.append((agg_id, rebuild_val_unit_col(valid_col_units, val_unit, kmap)))
    return new_list


def rebuild_from_col(valid_col_units, from_, kmap):
    """
    Args:

    Returns:
    """
    if from_ is None:
        return from_

    fn_proc = lambda x: rebuild_table_unit_col(valid_col_units, x, kmap)
    from_['table_units'] = [fn_proc(table_unit) for table_unit in from_['table_units']]
    from_['conds'] = rebuild_condition_col(valid_col_units, from_['conds'], kmap)
    return from_


def rebuild_group_by_col(valid_col_units, group_by, kmap):
    """
    Args:

    Returns:
    """
    if group_by is None:
        return group_by

    return [rebuild_col_unit_col(valid_col_units, col_unit, kmap) for col_unit in group_by]


def rebuild_order_by_col(valid_col_units, order_by, kmap):
    """
    Args:

    Returns:
    """
    if order_by is None or len(order_by) == 0:
        return order_by

    direction, val_units = order_by
    new_val_units = [(agg_id, rebuild_val_unit_col(valid_col_units, val_unit, kmap)) for agg_id, val_unit in val_units]
    return direction, new_val_units


def rebuild_sql_col(valid_col_units, sql, kmap):
    """
    Args:

    Returns:
    """
    if sql is None:
        return sql

    sql['select'] = rebuild_select_col(valid_col_units, sql['select'], kmap)
    sql['from'] = rebuild_from_col(valid_col_units, sql['from'], kmap)
    sql['where'] = rebuild_condition_col(valid_col_units, sql['where'], kmap)
    sql['groupBy'] = rebuild_group_by_col(valid_col_units, sql['groupBy'], kmap)
    sql['orderBy'] = rebuild_order_by_col(valid_col_units, sql['orderBy'], kmap)
    sql['having'] = rebuild_condition_col(valid_col_units, sql['having'], kmap)
    sql['intersect'] = rebuild_sql_col(valid_col_units, sql['intersect'], kmap)
    sql['except'] = rebuild_sql_col(valid_col_units, sql['except'], kmap)
    sql['union'] = rebuild_sql_col(valid_col_units, sql['union'], kmap)

    return sql


def build_foreign_key_map(entry):
    """
    Args:

    Returns:
    """
    cols_orig = entry["column_names"]
    tables_orig = entry["table_names"]

    # rebuild cols corresponding to idmap in Schema
    cols = []
    for col_orig in cols_orig:
        if col_orig[0] >= 0:
            t = tables_orig[col_orig[0]]
            c = col_orig[1]
            cols.append("__" + t.lower() + "." + c.lower() + "__")
        else:
            cols.append("__all__")

    def keyset_in_list(k1, k2, k_list):
        """keyset_in_list"""
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set

    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]

    return foreign_key_map


def build_foreign_key_map_from_json(table):
    """
    Args:

    Returns:
    """
    with open(table, encoding='utf-8') as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry['db_id']] = build_foreign_key_map(entry)
    return tables


def evaluate(table, gold, predict):
    """evaluate main

    Args:
        table (str): all tables file name
        gold (str): gold file name
        pred (str): predict file name

    Returns: float
        exact match acc
    """
    kmaps = build_foreign_key_map_from_json(table)

    with open(table, encoding='utf-8') as ifs:
        table_list = json.load(ifs)
        table_dict = {}
        for table in table_list:
            table_dict[table['db_id']] = table
    # with open(gold, encoding='utf-8') as ifs:
    #     gold_list = [l.strip().split('\t') for l in ifs if len(l.strip()) > 0]
    #     gold_dict = dict([(x[0], x[1:]) for x in gold_list])

    with open(gold, encoding='utf-8') as ifs:
        gold_list = json.load(ifs)

        gold_list = [ [l['question_id'], 
                       l['original_sql_query'] if 'original_sql_query' in l.keys() else l['sql_query'], 
                       l['db_id']] 
                       for l in gold_list]

        gold_dict = dict([(x[0], x[1:]) for x in gold_list])

    with open(predict, encoding='utf-8') as ifs:
        pred_list = [l.strip().split('\t') for l in ifs if len(l.strip()) > 0]
        pred_dict = dict([(x[0], x[1:]) for x in pred_list])

    evaluator = Evaluator()

    # partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
    #                  'group', 'order', 'and/or', 'IUEN', 'keywords']

    partial_types = ['select', 'where', 'group(no Having)', 'group', 'order', 'and/or', 'IUEN', 'keywords']

    scores = {
        'all': {'count': 0, 'exact': 0}
    }

    partial_scores = {}
    for type_ in partial_types:
        partial_scores[type_] = {'acc': 0., 'rec': 0., 'f1': 0., 'acc_count': 0, 'rec_count': 0}

    eval_err_num = 0
    for ins_id, g in gold_dict.items():
        scores['all']['count'] += 1
        if ins_id not in pred_dict:
            continue
        p = pred_dict[ins_id]
        if len(p) == 0:
            continue
        pred_str = p[0]
        gold_str, db_id = g
        schema = Schema(table_dict[db_id])

        gold_str = clean_sql_query(gold_str)
        gold_sql = get_sql(schema, gold_str)

        kmap = kmaps[db_id]
        # rebuild sql for value evaluation
        g_valid_col_units = build_valid_col_units(gold_sql['from']['table_units'], schema)
        gold_sql = rebuild_sql_col(g_valid_col_units, gold_sql, kmap)

        try:
            pred_str = clean_sql_query(pred_str)
            pred_sql = get_sql(schema, pred_str)

            p_valid_col_units = build_valid_col_units(pred_sql['from']['table_units'], schema)
            pred_sql = rebuild_sql_col(p_valid_col_units, pred_sql, kmap)
        except Exception as e:
            # If pred_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            pred_sql = g_empty_sql
            eval_err_num += 1

        exact_score = evaluator.eval_exact_match(pred_sql, gold_sql)
        partial_score = evaluator.partial_scores
        for type_ in partial_types:
            if partial_score[type_]['pred_total'] > 0:
                partial_scores[type_]['acc'] += partial_score[type_]['acc']
                partial_scores[type_]['acc_count'] += 1
            if partial_score[type_]['gold_total'] > 0:
                partial_scores[type_]['rec'] += partial_score[type_]['rec']
                partial_scores[type_]['rec_count'] += 1
            partial_scores[type_]['f1'] += partial_score[type_]['f1']

        #   print(res['keywords'])
        if exact_score == 0:
            logging.debug("error instance %s:\npred: %s\ngold: %s" % (ins_id, pred_str, gold_str))

        # else:
        #    print(gold_str)
        #    print(pred_str)
        scores['all']['exact'] += exact_score

    if scores['all']['count'] == 0:
        logging.warn('the number of evaluated instance is zero')
        return 0.0

    acc = scores['all']['exact'] / scores['all']['count']

    partial_acc = 0
    partial_rec = 0
    partial_f1 = 0

    for type_ in partial_types:
        if partial_scores[type_]['acc_count'] == 0:
            partial_scores[type_]['acc'] = 0
        else:
            partial_scores[type_]['acc'] = partial_scores[type_]['acc'] / \
                                           partial_scores[type_]['acc_count'] * 1.0
        partial_acc += partial_scores[type_]['acc']

        if partial_scores[type_]['rec_count'] == 0:
            partial_scores[type_]['rec'] = 0
        else:
            partial_scores[type_]['rec'] = partial_scores[type_]['rec'] / \
                                           partial_scores[type_]['rec_count'] * 1.0
        partial_rec += partial_scores[type_]['rec']

        if partial_scores[type_]['acc'] == 0 and partial_scores[type_]['rec'] == 0:
            partial_scores[type_]['f1'] = 0
        else:
            partial_scores[type_]['f1'] = \
                2.0 * partial_scores[type_]['acc'] * partial_scores[type_]['rec'] / (
                        partial_scores[type_]['rec'] + partial_scores[type_]['acc'])
        partial_f1 += partial_scores[type_]['f1']

    type_num = len(partial_types) * 1.0

    return acc, partial_acc / type_num, partial_rec / type_num, partial_f1 / type_num, partial_scores


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()

    arg_parser.add_argument('-g', '--gold', dest='gold', default='data/spdbV1.4/fold_1.4_1/gold_val_1.4.json')
    arg_parser.add_argument('-p', '--pred', dest='pred', default='logdir/duorat-electra-base-fold1-6layer/output-50000')
    arg_parser.add_argument('-s', '--schema', dest='schema', default='data/spdbV1.4/new_db_schema.json')
    args = arg_parser.parse_args()

    acc, part_acc, part_rec, part_f1, part_scores = evaluate(args.schema, args.gold, args.pred)


    print('============ ALL partial match ===============')
    print("exact acc: {:6f}, partial acc: {:6f}, partial rec: {:6f}, partial f1: {:6f}".format(acc, part_acc, part_rec,
                                                                                               part_f1))
    print('\n')

    for type in ['select', 'where', 'group(no Having)', 'group', 'order', 'and/or', 'IUEN', 'keywords']:
        print('============ {} partial match ==============='.format(type.upper()))

        print("partial acc: {:6f}, partial rec: {:6f}, partial f1: {:6f}".format(part_scores[type]['acc'],
                                                                                 part_scores[type]['rec'],
                                                                                 part_scores[type]['f1']))
        print('\n')
