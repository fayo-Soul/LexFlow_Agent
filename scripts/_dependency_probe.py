import os

import pymysql
import redis
from pymilvus import connections


mysql = pymysql.connect(
    host=os.environ["MYSQL_HOST"],
    user=os.environ["MYSQL_USER"],
    password=os.environ["MYSQL_PASSWORD"],
    database=os.environ["MYSQL_DATABASE"],
)
mysql.close()
print("mysql-ok")

cache = redis.Redis(
    host=os.environ["REDIS_HOST"],
    port=int(os.environ["REDIS_PORT"]),
)
assert cache.ping()
print("redis-ok")

connections.connect(
    host=os.environ["MILVUS_HOST"],
    port=os.environ["MILVUS_PORT"],
)
connections.disconnect("default")
print("milvus-ok")
