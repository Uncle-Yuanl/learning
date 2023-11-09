"""
尝试使用多线程进行数据上传
"""
import json
from tqdm import tqdm
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
from io import StringIO
import threading
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
import ctypes
libc = ctypes.cdll.LoadLibrary('libc.so.6')
SYS_gettid = 186

import sys
sys.path.append('/home/yhao/code/learning')
from utils import timer


class PGC:
    def __init__(self, host, dbname, schema):
        self.host = host
        self.dbname = dbname
        self.schema = schema
        self.conn = psycopg2.connect(
            host=host, port='8080', user='cmiai', password='cmisea19', 
            database=self.dbname, options=f'-c search_path={self.schema}',
            connect_timeout=600)

    def _upload_by_cpoy(self, df, table_name):
        """线程执行操作：通过cpoy_expert数据上传
        问题：
            线程之间存在线性等待的关系，并不能并行，可能是由于cpoy_expert的关系
        """
        flogger = logging.getLogger(f"【线程：{libc.syscall(SYS_gettid)}】")
        s_buf = StringIO()
        df.to_csv(s_buf, sep='\t', index=False)
        s_buf.seek(0)
        columns = ', '.join(['"{}"'.format(k) for k in df.columns])

        # IO操作
        # 错误: This bug is about an issue for which on python 3 processes closed each other's connections
        # with self.conn:
        conn = psycopg2.connect(
            host=self.host, port='8080', user='cmiai', password='cmisea19', 
            database=self.dbname, options=f'-c search_path={self.schema}',
            connect_timeout=600)
        with timer('该进程写入数据库', flogger):
            # cur = conn.cursor()
            with conn.cursor() as cur:
                sql = """COPY {} ({}) FROM STDIN DELIMITER '\t' CSV HEADER""".format(
                    table_name, columns)
                cur.copy_expert(sql=sql, file=s_buf)

    def _upload_by_insert(self, df, table_name):
        """线程执行操作：通过insert数据上传
        """
        flogger = logging.getLogger(f"【线程：{libc.syscall(SYS_gettid)}】")
        conn = psycopg2.connect(
            host=self.host, port='8080', user='cmiai', password='cmisea19', 
            database=self.dbname, options=f'-c search_path={self.schema}',
            connect_timeout=600)
        with timer('该进程写入数据库', flogger):
            with conn.cursor() as cur:
                sql = """
                    insert into {} values (%s, %s, %s, %s, %s, %s)
                """.format(table_name)
                cur.executemany(sql, tuple(df.apply(tuple, axis=1).values))
                conn.commit()

    def upload(self, df, engine, target_tablename, method, numthread=200, if_exists='append'):
        """多线程数据上传
        """
        pd_sql_engine = pd.io.sql.pandasSQL_builder(engine)
        table = pd.io.sql.SQLTable(
            target_tablename, 
            pd_sql_engine, 
            frame=df,
            index=False, 
            if_exists=if_exists, 
            schema=self.schema
            )
        if not table.exists():
            table.create()
        
        # 多线程写入
        with timer(f'多线程写入数据库', logger):
            numpert = len(df) // numthread + 1
            threads = [
                threading.Thread(
                    target=method,
                    args=(
                        df[t * numpert : (t + 1) * numpert],
                        target_tablename,
                    )
                ) for t in range(numthread)
            ]

            # 启动进程
            [t.start() for t in threads]
            # 阻塞进程
            [t.join() for t in threads]

    def s_(self, df, engine, target_tablename, if_exists='append'):
        pd_sql_engine = pd.io.sql.pandasSQL_builder(engine)
        table = pd.io.sql.SQLTable(
            target_tablename, 
            pd_sql_engine, 
            frame=df,
            index=False, 
            if_exists=if_exists, 
            schema=self.schema
            )
        if not table.exists():
            table.create()

        with timer('写入数据库', logging.getLogger('单进程')):
            with self.conn:
                with self.conn.cursor() as cur:
                    s_buf = StringIO()
                    df.to_csv(s_buf, sep='\t', index=False)  # header写进去，postgresql会自动删除第一条
                    s_buf.seek(0)
                    columns = ', '.join(['"{}"'.format(k) for k in df.columns])
                    if table.schema:
                        table_name = '{}.{}'.format(table.schema, table.name)
                    else:
                        table_name = table.name
                    sql = """COPY {} ({}) FROM STDIN DELIMITER '\t' CSV HEADER""".format(
                        table_name, columns)
                    cur.copy_expert(sql=sql, file=s_buf)


if __name__ == "__main__":
    numthread = 100
    logger = logging.getLogger(f"总线程数：{numthread}")
    engine_local = create_engine('postgresql://cmiai:cmisea19@localhost:8080/postgres')
    engine_dbhost = create_engine('postgresql://cmiai:cmisea19@dbhost:8080/review_db')

    read_sql = """
        select * from skincare_review_nlp_bak_transfer limit 1000000;
    """
    df = pd.read_sql(read_sql, engine_local)
    pgc = PGC('dbhost', 'review_db', 'staging')
    pgc.s_(df, engine_dbhost, 'thread_single')
    # pgc.upload(df, engine_dbhost, 'thread_multi', pgc._upload_by_insert)


