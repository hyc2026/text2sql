import re



SQL_KEYWORDS = ['select', 'from', 'by', 'where', 'as', 'group',
                'join', 'order', 'on', 'limit', 'avg', 'count',
                'having', 'asc', 'and', 'desc', 'sum', 'max', 'min',
                'or', 'a', 'b', 'in', 'except', 'union', 'not',
                'like', 'intersect']

def is_keyword(word):
    word = word.lower()
    return word in SQL_KEYWORDS


# 关键字转小写
def keywords_to_lower(sql):
    words = sql.split()
    new_sql = [w.lower() if is_keyword(w) else w for w in words]
    new_sql = ' '.join(new_sql)
    return new_sql

def clean_name(str1):
    char_for_trans = ['—', '-', '=', '－', '.', ')', '(', '，', '/', ':', ' ', '―']
    for char in char_for_trans:
        str1 = str1.replace(char, '_')
    return str1


def change_under_line(str1):
    # 使用非贪婪匹配查找query中的列名、表名
    r = r'\".+?\"'
    result = re.findall(r, str1)
    if len(result) > 0:
        for name_str in result:
            # 去掉query中列名表名的符号
            clean_name_str = clean_name(name_str)
            str1 = str1.replace(name_str, clean_name_str)

    return str1


def clean_sql_query(query):
    query = change_under_line(query)
    # DuSQL 中 ( )左右都是用空格的，如 count ( * )
    query = query.replace('(', ' ( ')
    query = query.replace(')', ' ) ')
    # DuSQL 中 , 左右都是有空格的，如 select a , b
    query = query.replace(',', ' , ')
    # 打碎重构，去掉多余空格
    query = ' '.join([i for i in query.split()])
    # DuSQL中只有 where 的条件中的 value 为字符串时才会有 单引号，其他都没有
    query = query.replace('\"', '').replace(' = ', ' == ')
    # 将 query 中的 time_now 小写转大写
    query = query.replace('time_now', 'TIME_NOW')
    # DuSQL 中 where 里的时间是没有单引号的，如 where 更新时间 > 2012-12-05
    # 转换 where 更新时间 > '2012-12-05' 为 where 更新时间 > 2012-12-05
    # 下面几行代码很 magic，不要错过哦
    result = re.findall('\d+-\d+-\d+', query)
    if len(result) > 0:
        for date_str in result:
            # 去掉日期的单引号
            query = query.replace(f'\'{date_str}\'', date_str)

    # 只有关键字需要转小写
    query = keywords_to_lower(query)

    return query
