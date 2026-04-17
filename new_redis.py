"""
new_redis.py - 生产级 Redis 缓存管理器
解决：Big Key、连接池单例、缓存击穿、缓存雪崩
"""

import redis
import json
import hashlib
import logging
import random
import time
import uuid
from typing import Optional, Callable

logger = logging.getLogger("RedisManager")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class RedisClientWrapper:
    _pool = None

    def __init__(self, host='0.0.0.0', port=6379, db=0, password=None, max_connections=100):
        if not RedisClientWrapper._pool:
            RedisClientWrapper._pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
        self.client = redis.StrictRedis(connection_pool=RedisClientWrapper._pool)

        # 预加载 Lua 脚本（原子性释放锁）
        self.unlock_script = self.client.register_script("""
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
        """)

        try:
            self.client.ping()
            logger.info("Redis Connected Successfully")
        except redis.ConnectionError:
            logger.error("Redis Connection Failed")

    def _generate_key(self, text: str, prefix: str = "llm:cache:") -> str:
        hash_obj = hashlib.md5(text.encode('utf-8'))
        return f"{prefix}{hash_obj.hexdigest()}"

    def get_answer(self, question: str) -> Optional[str]:
        key = self._generate_key(question)
        try:
            val = self.client.get(key)
            if val:
                logger.info(f"Cache Hit: {key}")
                if val == "<EMPTY>":
                    return None
                return val
        except redis.RedisError as e:
            logger.error(f"Redis Read Error: {e}")
        return None

    def set_answer(self, question: str, answer: str, expire_time: int = 3600):
        key = self._generate_key(question)
        jitter = random.randint(int(-expire_time * 0.1), int(expire_time * 0.1))
        real_expire = expire_time + jitter
        try:
            self.client.setex(key, real_expire, answer)
        except redis.RedisError as e:
            logger.error(f"Redis Write Error: {e}")

    def acquire_lock(self, lock_name: str, acquire_timeout=3, lock_timeout=10) -> Optional[str]:
        identifier = str(uuid.uuid4())
        lock_key = f"lock:{lock_name}"
        end = time.time() + acquire_timeout
        while time.time() < end:
            if self.client.set(lock_key, identifier, ex=lock_timeout, nx=True):
                return identifier
            time.sleep(0.01)
        return None

    def release_lock(self, lock_name: str, identifier: str) -> bool:
        lock_key = f"lock:{lock_name}"
        try:
            result = self.unlock_script(keys=[lock_key], args=[identifier])
            return bool(result)
        except redis.RedisError as e:
            logger.error(f"Lock Release Error: {e}")
            return False

    def get_or_compute(self, question: str, compute_func: Callable[[], str]) -> str:
        cached_ans = self.get_answer(question)
        if cached_ans:
            print('REDIS HIT !!!')
            return cached_ans

        hash_key = hashlib.md5(question.encode('utf-8')).hexdigest()
        lock_token = self.acquire_lock(hash_key)
        if lock_token:
            try:
                # Double Check
                cached_ans_retry = self.get_answer(question)
                if cached_ans_retry:
                    print('REDIS HIT (Double Check) !!!')
                    return cached_ans_retry
                print("Cache Miss, Computing LLM...")
                answer = compute_func()
                if answer:
                    self.set_answer(question, answer)
                else:
                    self.client.setex(self._generate_key(question), 60, "<EMPTY>")
                return answer
            finally:
                self.release_lock(hash_key, lock_token)
        else:
            time.sleep(0.1)
            return self.get_answer(question) or "System busy, calculating..."


# 单例模式初始化
redis_manager = RedisClientWrapper()
