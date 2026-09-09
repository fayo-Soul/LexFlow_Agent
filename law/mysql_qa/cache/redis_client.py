# 1.导包
import redis
# 由于我们要保持在redis中的数据可能是列表、字典等格式，所以需要引入json模块，用于序列化(json格式字符串)
import json
from base import setup_logger,Config
# 2.封装RedisClient类，用于实现Redis增删改查操作
class RedisClient(object):
    def __init__(self):
        # 2.1创建日志对象
        self.logger=setup_logger("Redis")
        # 2.2创建Redis连接
        try:
            self.client=redis.StrictRedis(
                host=Config().REDIS_HOST,
                port=Config().REDIS_PORT,
                db=Config().REDIS_DB,
                password=Config().REDIS_PASSWORD,
                # 默认情况下，Redis返回的都是字节，需要设置decode_responses=True，将字节转为字符串
                decode_responses=True
            )
            # 添加ping测试代码
            self.client.ping()
            self.logger.info(f'Redis连接成功')
        except redis.RedisError as e:
            self.logger.error(f'Redis连接失败:{e}')
            raise
    # 3. 设置key:value键值对
    def set_data(self,key,value):
        try:
            self.client.set(key,json.dumps(value,ensure_ascii=False))
            item_count = len(value) if hasattr(value, "__len__") else 1
            self.logger.info(f'数据写入成功：key={key}, item_count={item_count}')
        except Exception as e:
            self.logger.error(f'数据写入失败：{e}')
            raise
    # 4. 获取key对应的value
    def get_data(self,key):
        try:
            data=self.client.get(key)
            value = json.loads(data) if data else None
            item_count = len(value) if hasattr(value, "__len__") else (1 if value is not None else 0)
            self.logger.info(f'数据获取成功：key={key}, item_count={item_count}')
            return value
        except Exception as e:
            self.logger.error(f'数据获取失败：{e}')
            raise
    # 5.封装一个get_answer函数，用于获取答案
    def get_answer(self,query):
        try:
            answer=self.client.get(f'answer:{query}')
            if answer:
                self.logger.info('从Redis中成功获取缓存答案')
                return answer
            # 如果answer为None代表没有与之对于的答案，则返回None
            return None
        except redis.RedisError as e:
            self.logger.error(f'从Redis中获取答案失败：{e}')
            return None
if __name__ == '__main__':
    redis_client=RedisClient()
    # 提前录入数据到Redis
    # redis_client.set_data('answer:Python中列表和元素区别','列表和元组都是可变对象，列表可以进行增删改查，元组不可以进行增删改查')
    # 用户输入问题
    query="Python中列表和元素区别"
    # 获取答案
    result=redis_client.get_answer(query)
    print(result)
