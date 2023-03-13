
import collections
import itertools
import os

import asdl
import attr
import networkx as nx

from duorat.asdl.lang.dusql import ast_util
from duorat.datasets.spider import SpiderSchema
NO_ADD = 0
ADD_TALBE = 1
ADD_ALIAS = 2

def bimap(first, second):
    # 返回两个字典，first[i] -> second[i] 和 second[i] -> first[i]
    return {f: s for f, s in zip(first, second)}, {s: f for f, s in zip(first, second)}


def filter_nones(d):
    return {k: v for k, v in d.items() if v is not None and v != []}


def join(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x


def intersperse(delimiter, seq):
    return itertools.islice(
        itertools.chain.from_iterable(zip(itertools.repeat(delimiter), seq)), 1, None
    )


class DusqlGrammar:

    root_type = "sql"

    # bimap 返回两个字典，first[i] -> second[i] 和 second[i] -> first[i]
    COND_TYPES_F, COND_TYPES_B = bimap(
        # ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists'),
        # (None, 'Between', 'Eq', 'Gt', 'Lt', 'Ge', 'Le', 'Ne', 'In', 'Like', 'Is', 'Exists'))
        # dusql: (not_in, between, =, >, <, >=, <=, !=, in, like)
        range(0, 10), ("NI", "Between", "Eq", "Gt", "Lt", "Ge", "Le", "Ne", "In", "Like"),
    )

    UNIT_TYPES_F, UNIT_TYPES_B = bimap(
        # ('none', '-', '+', '*', '/'),
        range(5), ("Column", "Minus", "Plus", "Times", "Divide"),
    )

    AGG_TYPES_F, AGG_TYPES_B = bimap(
        range(6), ("NoneAggOp", "Max", "Min", "Count", "Sum", "Avg")
    )

    ORDERS_F, ORDERS_B = bimap(("asc", "desc"), ("Asc", "Desc"))

    LOGIC_OPERATORS_F, LOGIC_OPERATORS_B = bimap(("and", "or"), ("And", "Or"))

    def __init__(
        self,
        output_from=False,  # 是否输出 from 关键字
        use_table_pointer=False,
        include_literals=True,
        include_columns=True,
    ):
        custom_primitive_type_checkers = {}

        # self.pointers 用于以指针网络方式来挑选 table 和 column
        self.pointers = set()

        # 如果要以指针网络方式来挑选 table 数据表
        if use_table_pointer:
            custom_primitive_type_checkers["table"] = ast_util.FilterType(int)
            self.pointers.add("table")

        # 如果要以指针网络方式来挑选 column 数据列
        if include_columns:
            custom_primitive_type_checkers["column"] = ast_util.FilterType(int)
            self.pointers.add("column")

        # TODO (YuweiYin) 对 ast_util.ASTWrapper 类进行注解
        self.ast_wrapper = ast_util.ASTWrapper(
            # 调用 asdl 软件包的解析过程 Parse ASDL from the given file and return a Module node describing it
            # with open(filename) as f: return ASDLParser().parse(f.read())
            asdl.parse(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dusql.asdl")
            ),
            custom_primitive_type_checkers=custom_primitive_type_checkers,
        )
        self.output_from = output_from
        self.include_literals = include_literals
        self.include_columns = include_columns

        # 如果不用输出 from 关键字，则删除之
        if not self.output_from:
            sql_fields = self.ast_wrapper.product_types["sql"].fields
            assert sql_fields[1].name == "from"
            del sql_fields[1]

        # 如果不以指针网络方式来挑选 table 数据表
        # SingularType = Union[asdl.Constructor, asdl.Product]
        if not use_table_pointer:
            self.ast_wrapper.singular_types["Table"].fields[0].type = "int"

        # TODO (YuweiYin) 解释 include_literals 含义
        if not include_literals:
            sql_fields = self.ast_wrapper.singular_types["sql"].fields
            for field in sql_fields:
                if field.name == "limit":
                    field.opt = False
                    field.type = "singleton"
        # TODO (YuweiYin) 解释 include_columns 含义
        if not include_columns:
            col_unit_fields = self.ast_wrapper.singular_types["col_unit"].fields
            assert col_unit_fields[1].name == "col_id"
            del col_unit_fields[1]

    def parse(self, code: dict):
        # 解析源数据中的 "sql" 字段 (dict)
        return self.parse_sql(code)

    def unparse(self, tree, spider_schema: SpiderSchema):
        # 反向解析 (parse 的逆过程)
        unparser = DusqlUnparser(self.ast_wrapper, spider_schema)
        return unparser.unparse_sql(tree)

    @classmethod
    def tokenize_field_value(cls, field_value):
        """对某个域/字段 的值进行编码，在 decoder.py 和 spider_beam_search.py 这两个生成过程中会使用到"""
        if isinstance(field_value, bytes):
            # 如果是字节流，则用 "latin1" 格式编码 TODO
            field_value_str = field_value.encode("latin1")
        elif isinstance(field_value, str):
            # 如果是字符串，则保持原样
            field_value_str = field_value
        else:
            # 否则转为字符串对象
            field_value_str = str(field_value)
            # 如果首尾有双引号，则去掉之
            if field_value_str[0] == '"' and field_value_str[-1] == '"':
                field_value_str = field_value_str[1:-1]
        # TODO: Get rid of surrounding quotes
        return [field_value_str]

    # parse 函数解析：每一个字典中
    # "_type" 与 asdl 中的 construct name 相互对应
    # 其余属性对应该 construct 所拥有的所有 field

    # sql语句dusql与spider相同
    def parse_sql(self, sql, optional=False):
        # 递归解析 sql 结构
        if optional and sql is None:
            return None
        # print(sql['select'])
        return filter_nones(
            {
                "_type": "sql",
                "select": self.parse_select(sql["select"]),
                "where": self.parse_cond(sql["where"], optional=True),
                "group_by": [self.parse_col_unit(u) for u in sql["groupBy"]],
                "order_by": self.parse_order_by(sql["orderBy"]),
                "having": self.parse_cond(sql["having"], optional=True),
                "limit": sql["limit"]
                if self.include_literals
                else (sql["limit"] is not None),
                # 递归 self.parse_sql
                "intersect": self.parse_sql(sql["intersect"], optional=True),
                "except": self.parse_sql(sql["except"], optional=True),
                "union": self.parse_sql(sql["union"], optional=True),
                **({"from": self.parse_from(sql["from"]),} if self.output_from else {}),
            }
        )

    # from语句dusql与spider相同
    def parse_from(self, from_):
        """解析 from 子句"""
        return filter_nones(
            {
                "_type": "from",
                "table_units": [self.parse_table_unit(u) for u in from_["table_units"]],
                "conds": self.parse_cond(from_["conds"], optional=True),
            }
        )


    # 修改val_untis为aggs
    def parse_order_by(self, order_by):
        """解析 order by 排序子句"""
        # dusql 和 spider 有区别，dusql 多一个 agg_id
        if not order_by:
            return None

        order, aggs = order_by
        # print(order_by)
        # print(order, val_units)
        return {
            "_type": "order_by",
            "order": {"_type": self.ORDERS_F[order]},
            "aggs": [self.parse_agg(agg) for agg in aggs],
        }

    def parse_cond(self, cond, optional=False):
        """解析 cond 判断条件"""
        if optional and not cond:
            return None

        if len(cond) > 1:
            return {
                "_type": self.LOGIC_OPERATORS_F[cond[1]],
                "left": self.parse_cond(cond[:1]),
                "right": self.parse_cond(cond[2:]),
            }

        ((agg_id, op_id, val_unit, val1, val2),) = cond

        agg = (agg_id, val_unit)
        result = {
            "_type": self.COND_TYPES_F[op_id],
            "aggs": self.parse_agg(agg),
            "val1": self.parse_val(val1),
        }
        """
        result = {
            "_type": self.COND_TYPES_F[op_id],
            "val_unit": self.parse_val_unit(val_unit),
            "val1": self.parse_val(val1),
        }
        """
        if op_id == 1:  # between
            result["val2"] = self.parse_val(val2)

        return result

    def parse_val(self, val):
        """解析 val 具体的值"""
        if isinstance(val, str):
            if not self.include_literals:
                return {"_type": "Terminal"}
            return {
                "_type": "String",
                "s": val,
            }
        elif isinstance(val, list):
            return {
                "_type": "ColUnit",
                "c": self.parse_col_unit(val),
            }
        elif isinstance(val, float) or isinstance(val, int):
            if not self.include_literals:
                return {"_type": "Terminal"}
            return {
                "_type": "Number",
                "f": float(val),
            }
        elif isinstance(val, dict):
            return {
                "_type": "ValSql",
                "s": self.parse_sql(val),
            }
        else:
            raise ValueError(val)

    # 删除distinct
    def parse_select(self, select):
        """解析 select 选择子句"""
        # duosql 里没有 distinct
        aggs = select
        return {
            "_type": "select",
            "aggs": [self.parse_agg(agg) for agg in aggs],
        }

    def parse_agg(self, agg):
        """解析 agg 聚合函数"""
        agg_id, val_unit = agg
        return {
            "_type": "agg",
            "agg_id": {"_type": self.AGG_TYPES_F[agg_id]},
            "val_unit": self.parse_val_unit(val_unit),
        }

    def parse_val_unit(self, val_unit):
        """解析 val_unit 值单元"""
        # print(val_unit)
        unit_op, col_unit1, col_unit2 = val_unit
        result = {
            "_type": self.UNIT_TYPES_F[unit_op],
            "col_unit1": self.parse_col_unit(col_unit1),
        }
        if unit_op != 0:
            result["col_unit2"] = self.parse_col_unit(col_unit2)
        return result

    #dusql比spider少了distinct，删除distinct属性
    def parse_col_unit(self, col_unit):
        """解析 col_unit 列单元"""
        # dusql 里没有 distinct
        agg_id, col_id = col_unit
        result = {
            "_type": "col_unit",
            "agg_id": {"_type": self.AGG_TYPES_F[agg_id]},
        }
        if self.include_columns:
            result["col_id"] = col_id
        return result

    def parse_table_unit(self, table_unit):
        """解析 order by 排序子句"""
        table_type, value = table_unit
        if table_type == "sql":
            return {
                "_type": "TableUnitSql",
                "s": self.parse_sql(value),
            }
        elif table_type == "table_unit":
            return {
                "_type": "Table",
                "table_id": value,
            }
        else:
            raise ValueError(table_type)


@attr.s
class DusqlUnparser:
    # 反向解析 (DusqlGrammar.parse 的逆过程)
    ast_wrapper = attr.ib()
    schema = attr.ib()

    UNIT_TYPES_B = {
        "Minus": "-",
        "Plus": "+",
        "Times": "*",
        "Divide": "/",
    }
    COND_TYPES_B = {
        "NI":"NOT IN",
        "Between": "BETWEEN",
        "Eq": "==",
        "Gt": ">",
        "Lt": "<",
        "Ge": ">=",
        "Le": "<=",
        "Ne": "!=",
        "In": "IN",
        "Like": "LIKE",
    }

    @classmethod
    def conjoin_conds(cls, conds):
        if not conds:
            return None
        if len(conds) == 1:
            return conds[0]
        return {"_type": "And", "left": conds[0], "right": cls.conjoin_conds(conds[1:])}

    @classmethod
    def linearize_cond(cls, cond):
        if cond["_type"] in ("And", "Or"):
            conds, keywords = cls.linearize_cond(cond["right"])
            return [cond["left"]] + conds, [cond["_type"]] + keywords
        else:
            return [cond], []

    def unparse_val(self, val, add_table_name=NO_ADD):
        if val["_type"] == "Terminal":
            return "'terminal'"
        if val["_type"] == "String":
            res = val["s"].replace(' ', "")
            res = '\'{}\''.format(res)
            return res
        if val["_type"] == "ColUnit":
            return self.unparse_col_unit(val["c"], add_table_name)
        if val["_type"] == "Number":
            return str(val["f"])
        if val["_type"] == "ValSql":
            return "({})".format(self.unparse_sql(val["s"]))

    def unparse_col_unit(self, col_unit, add_table_name=NO_ADD):
        if "col_id" in col_unit:
            try:
                column = self.schema.columns[col_unit["col_id"]]
            except IndexError:
                column = self.schema.columns[0]
            if column.table is None:
                column_name = column.orig_name_for_unparse
            else:
                if add_table_name == NO_ADD:
                    column_name = "\"{}\"".format(column.orig_name_for_unparse)
                elif add_table_name == ADD_TALBE:
                    column_name = "\"{}\".\"{}\"".format(column.table.orig_name_for_unparse, column.orig_name_for_unparse)
                else:
                    column_name = "{}.\"{}\"".format(add_table_name, column.orig_name_for_unparse)
                #column_name = "{}.{}".format(column.table.orig_name, column.orig_name)
                #column_name = "{}".format(column.orig_name)
        else:
            column_name = "some_col"

        if column_name.find('当前时间') != -1:
            column_name = '\"TIME_NOW\"'
        agg_type = col_unit["agg_id"]["_type"]
        if column_name in  ['\"*\"']:
            column_name = " * "
        else:
            column_name = " {} ".format(column_name)
        if agg_type == "NoneAggOp":
            return column_name
        else:
            return "{}({})".format(agg_type, column_name)

    def unparse_val_unit(self, val_unit, add_table_name=NO_ADD):
        # 如果要加别名，说明肯定是select中出现的，根据spdb数据，第一个col_init为a，第二个col_unit为b
        if val_unit["_type"] == "Column":
            return self.unparse_col_unit(val_unit["col_unit1"], add_table_name if add_table_name != ADD_ALIAS else "a")
        col1 = self.unparse_col_unit(val_unit["col_unit1"], add_table_name if add_table_name != ADD_ALIAS else "a")
        col2 = self.unparse_col_unit(val_unit["col_unit2"], add_table_name if add_table_name != ADD_ALIAS else "b")
        return "{} {} {}".format(col1, self.UNIT_TYPES_B[val_unit["_type"]], col2)

    # def unparse_table_unit(self, table_unit):
    #    raise NotImplementedError

    def unparse_cond(self, cond, add_table_name=NO_ADD, negated=False):
        if cond["_type"] == "And":
            assert not negated
            return "{} AND {}".format(
                self.unparse_cond(cond["left"], add_table_name), self.unparse_cond(cond["right"], add_table_name)
            )
        elif cond["_type"] == "Or":
            assert not negated
            return "{} OR {}".format(
                self.unparse_cond(cond["left"], add_table_name), self.unparse_cond(cond["right"], add_table_name)
            )
        elif cond["_type"] == "Between":
            tokens = [self.unparse_agg(cond["aggs"], add_table_name)]
            if negated:
                tokens.append("NOT")
            tokens += [
                "BETWEEN",
                self.unparse_val(cond["val1"], add_table_name),
                "AND",
                self.unparse_val(cond["val2"], add_table_name),
            ]
            return " ".join(tokens)
        tokens = [self.unparse_agg(cond["aggs"], add_table_name)]
        if negated:
            tokens.append("NOT")
        tokens += [self.COND_TYPES_B[cond["_type"]], self.unparse_val(cond["val1"], add_table_name)]
        return " ".join(tokens)

    def unparse_sql(self, tree):
        # First, fix 'from'
        if "from" not in tree:
            tree = dict(tree)

            # Get all candidate columns
            candidate_column_ids = set(
                self.ast_wrapper.find_all_descendants_of_type(
                    tree, "column", lambda field: field.type != "sql"
                )
            )
            candidate_columns = [self.schema.columns[i] for i in candidate_column_ids]
            all_from_table_ids = set(
                column.table.id
                for column in candidate_columns
                if column.table is not None
            )
            if not all_from_table_ids:
                # Copied from SyntaxSQLNet
                all_from_table_ids = {0}

            covered_tables = set()
            candidate_table_ids = sorted(all_from_table_ids)
            start_table_id = candidate_table_ids[0]
            conds = []
            for table_id in candidate_table_ids[1:]:
                if table_id in covered_tables:
                    continue
                try:
                    path = nx.shortest_path(
                        self.schema.foreign_key_graph,
                        source=start_table_id,
                        target=table_id,
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    covered_tables.add(table_id)
                    continue

                for source_table_id, target_table_id in zip(path, path[1:]):
                    if target_table_id in covered_tables:
                        continue
                    all_from_table_ids.add(target_table_id)
                    col1, col2 = self.schema.foreign_key_graph[source_table_id][
                        target_table_id
                    ]["columns"]
                    conds.append(
                        {
                            "_type": "Eq",
                            'aggs': {
                                '_type': 'agg',
                                'agg_id': {'_type': 'NoneAggOp'},
                                'val_unit': {
                                    '_type': 'Column',
                                    'col_unit1': {
                                        '_type': 'col_unit',
                                        'agg_id': {'_type': 'NoneAggOp'},
                                        'col_id': col1
                                    },
                                },
                            },
                            "val1": {
                                "_type": "ColUnit",
                                "c": {
                                    "_type": "col_unit",
                                    "agg_id": {"_type": "NoneAggOp"},
                                    "col_id": col2,
                                },
                            },
                        }
                    )
            table_units = [
                {"_type": "Table", "table_id": i} for i in sorted(all_from_table_ids)
            ]

            tree["from"] = {
                "_type": "from",
                "table_units": table_units,
            }
            cond_node = self.conjoin_conds(conds)
            if cond_node is not None:
                tree["from"]["conds"] = cond_node
        
        # 根据from子句中表的个数和类型，来判断是否需要加表名或别名
        add_table_type = NO_ADD
        if (len(tree["from"]["table_units"]) > 1):
            add_table_type = ADD_TALBE
        
        # 观察SPDB数据发现，所有子句都可能出现"表名.列名"，但只有select中会出现"表的别名.列名"，因此特判，只有在select子句中会使用表的别名
        select_table_type = add_table_type
        if (add_table_type == ADD_TALBE):
            # sql类型则需要加别名,否则加表名
            if tree["from"]["table_units"][0]["_type"] == "TableUnitSql":
                select_table_type = ADD_ALIAS            


        result = [
            # select select,
            self.unparse_select(tree["select"], select_table_type),
            # from from,
            self.unparse_from(tree["from"], add_table_type),
        ]
        # cond? where,
        if "where" in tree:
            result += ["WHERE", self.unparse_cond(tree["where"], add_table_type)]
        # col_unit* group_by,
        if "group_by" in tree:
            result += [
                "GROUP BY",
                ", ".join(self.unparse_col_unit(c, add_table_type) for c in tree["group_by"]),
            ]
        # cond? having,
        if "having" in tree:
            result += ["HAVING", self.unparse_cond(tree["having"], add_table_type)]
        # order_by? order_by,
        if "order_by" in tree:
            result.append(self.unparse_order_by(tree["order_by"], add_table_type))
        # int? limit,
        if "limit" in tree:
            if isinstance(tree["limit"], bool):
                if tree["limit"]:
                    result += ["LIMIT", "1"]
            else:
                result += ["LIMIT", str(tree["limit"])]

        # sql? intersect,
        if "intersect" in tree:
            result = ['(']+result+[')']
            result += ["INTERSECT", '(', self.unparse_sql(tree["intersect"]), ')']
        # sql? except,
        if "except" in tree:
            result = ['(']+result+[')']
            result += ["EXCEPT", '(', self.unparse_sql(tree["except"]), ')']
        # sql? union
        if "union" in tree:
            result = ['(']+result+[')']
            result += ["UNION", '(', self.unparse_sql(tree["union"]), ')']

        return " ".join(result)

    def unparse_select(self, select, add_table_name=NO_ADD):
        tokens = ["SELECT"]
        tokens.append(
            ", ".join(self.unparse_agg(agg, add_table_name) for agg in select.get("aggs", []))
        )
        return " ".join(tokens)

    def unparse_agg(self, agg, add_table_name=NO_ADD):
        unparsed_val_unit = self.unparse_val_unit(agg["val_unit"], add_table_name)
        agg_type = agg["agg_id"]["_type"]
        if agg_type == "NoneAggOp":
            return unparsed_val_unit
        else:
            return "{}({})".format(agg_type.upper(), unparsed_val_unit)

    def unparse_from(self, from_, add_table_name=NO_ADD):
        if "conds" in from_:
            all_conds, keywords = self.linearize_cond(from_["conds"])
        else:
            all_conds, keywords = [], []
        assert all(keyword == "And" for keyword in keywords)

        cond_indices_by_table = collections.defaultdict(set)
        tables_involved_by_cond_idx = collections.defaultdict(set)
        for i, cond in enumerate(all_conds):
            for column in self.ast_wrapper.find_all_descendants_of_type(cond, "column"):
                table = self.schema.columns[column].table
                if table is None:
                    continue
                cond_indices_by_table[table.id].add(i)
                tables_involved_by_cond_idx[i].add(table.id)

        output_table_ids = set()
        output_cond_indices = set()
        tokens = ["FROM"]
        for i, table_unit in enumerate(from_.get("table_units", [])):
            if i > 0 and table_unit["_type"] == "Table":
                tokens += ["JOIN"]

            if table_unit["_type"] == "TableUnitSql":
                tokens.append("({})".format(self.unparse_sql(table_unit["s"])))
                tokens += [chr(i+97)]
                if i < len(from_.get("table_units", [])) - 1:
                    tokens += [","]    
            elif table_unit["_type"] == "Table":
                table_id = table_unit["table_id"]
                temp = self.schema.tables[table_id].orig_name_for_unparse
                temp = ' \"{}\" '.format(temp)
                tokens += [temp]
                output_table_ids.add(table_id)

                # Output "ON <cond>" if all tables involved in the condition have been output
                conds_to_output = []
                for cond_idx in sorted(cond_indices_by_table[table_id]):
                    if cond_idx in output_cond_indices:
                        continue
                    if tables_involved_by_cond_idx[cond_idx] <= output_table_ids:
                        conds_to_output.append(all_conds[cond_idx])
                        output_cond_indices.add(cond_idx)
                if conds_to_output:
                    tokens += ["ON"]
                    tokens += list(
                        intersperse(
                            "AND", (self.unparse_cond(cond, add_table_name) for cond in conds_to_output)
                        )
                    )

            
        return " ".join(tokens)

    def unparse_order_by(self, order_by, add_table_name=NO_ADD):
        # 'val_units' has sequential cardinality (*) in the grammar, therefore it could be absent from order_by
        if "aggs" in order_by:
            return "ORDER BY {} {}".format(
                ", ".join(self.unparse_agg(v, add_table_name) for v in order_by["aggs"]),
                order_by["order"]["_type"].upper(),
            )
        else:
            return "ORDER BY {}".format(order_by["order"]["_type"].upper())
