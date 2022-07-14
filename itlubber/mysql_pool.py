import pymysql
import pandas as pd
from DBUtils.PooledDB import PooledDB
from .utils import get_logger


mysql_connect_pool_options = dict(
    creator=pymysql,
    host="itlubber.art",
    database="itlubber.art",
    user="itlubber.art",
    password="itlubber.art",
    port=3306,
    maxconnections=3,
    mincached=1,
    maxcached=0,
    maxshared=0,
    blocking=True,
    maxusage=None,
    setsession=[],
    ping=1,
    charset='utf8'
)


class MysqlPool(object):
    _logger = None
    _pool = None

    def __init__(self):
        self._logger = get_logger()
        self._pool = PooledDB(**mysql_connect_pool_options)

    def register_connect(self):
        _conn = self._pool.connection()
        _cursor = _conn.cursor(cursor=pymysql.cursors.DictCursor)
        return _conn, _cursor

    @staticmethod
    def logout_connect(_conn, _cursor):
        """关闭数据库连接"""
        try:
            _cursor.close()
            _conn.close()
        except Exception as e:
            self._logger.info("放回数据库连接池失败。")

    def query_dict(self, sql):
        _conn, _cursor = self.register_connect()
        query_dict_res = ''
        try:
            # 执行sql语句
            _cursor.execute(sql)
            query_dict_res = _cursor.fetchall()
        except Exception as e:
            self._logger.error('发生异常:', e)
        self.logout_connect(_conn, _cursor)
        return query_dict_res

    def query_data(self, sql):
        _conn, _cursor = self.register_connect()
        query_dict_res = ''
        try:
            # 执行sql语句
            _cursor.execute(sql)
            query_dict_res = pd.DataFrame(_cursor.fetchall())
        except Exception as e:
            self._logger.error('发生异常:', e)
        self.logout_connect(_conn, _cursor)
        return query_dict_res

    def execute_sql(self, sql):
        _conn, _cursor = self.register_connect()
        execute_res = False
        try:
            # 执行SQL语句
            _cursor.execute(sql)
            # 提交到数据库执行
            _conn.commit()
            execute_res = True
        except Exception as e:
            self._logger.error('发生异常:', e)
            # 发生错误时回滚
            _conn.rollback()
        self.logout_connect(_conn, _cursor)
        return execute_res


if __name__ == '__main__':
    mysql_pool = MysqlPool(mysql_connect_pool_options)
