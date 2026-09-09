#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法律问答对的 MySQL 客户端
表名：law_qa（字段：id, question, answer, domain, status, hit_count, created_by, create_time, update_time）
"""

import json
import pymysql
from base import setup_logger, Config


class LawMySQLClient:
    def __init__(self, table_name=None):
        self.table_name = table_name or Config().MYSQL_LAW_QA_TABLE
        self.logger = setup_logger('LawMySQL')
        try:
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,
                user=Config().MYSQL_USER,
                password=Config().MYSQL_PASSWORD,
                database=Config().MYSQL_DATABASE,
                charset='utf8mb4'
            )
            self.cursor = self.connection.cursor()
            self.logger.info('MySQL 连接成功')
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL 连接失败：{e}")
            raise

    def create_table(self):
        """创建或更新 law_qa 表（含企业级字段）。"""
        create_table_query = f'''
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            domain VARCHAR(50) DEFAULT '',
            status VARCHAR(20) DEFAULT "enabled",
            hit_count INT DEFAULT 0,
            created_by VARCHAR(50) DEFAULT "",
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        '''
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            self.logger.info(f'{self.table_name} 表创建成功')
        except pymysql.MySQLError as e:
            self.logger.error(f"数据表创建失败：{e}")
            raise

    def _upgrade_table_if_needed(self):
        """迁移旧表：新增企业字段（幂等）。"""
        try:
            self.cursor.execute(f"SHOW COLUMNS FROM {self.table_name} LIKE 'status'")
            if not self.cursor.fetchone():
                self.cursor.execute(f"ALTER TABLE {self.table_name} "
                                    "ADD COLUMN status VARCHAR(20) DEFAULT 'enabled' AFTER domain")
                self.cursor.execute(f"ALTER TABLE {self.table_name} "
                                    "ADD COLUMN hit_count INT DEFAULT 0 AFTER status")
                self.cursor.execute(f"ALTER TABLE {self.table_name} "
                                    "ADD COLUMN created_by VARCHAR(50) DEFAULT '' AFTER hit_count")
                self.cursor.execute(f"ALTER TABLE {self.table_name} "
                                    "ADD COLUMN update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP "
                                    "ON UPDATE CURRENT_TIMESTAMP AFTER create_time")
                self.connection.commit()
                self.logger.info(f'{self.table_name} 表结构已升级')
        except Exception as e:
            self.logger.warning(f"表结构升级跳过（可能已是最新）：{e}")

    def insert_data(self, jsonl_path):
        inserted = 0
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    insert_query = f"""
                        INSERT INTO {self.table_name} (question, answer, domain)
                        VALUES (%s, %s, %s)
                    """
                    self.cursor.execute(insert_query, (
                        item.get('question', ''),
                        item.get('answer', ''),
                        item.get('domain', '')
                    ))
                    inserted += 1
                    if inserted % 500 == 0:
                        self.connection.commit()
                        self.logger.info(f"已插入 {inserted} 条数据")
            self.connection.commit()
            self.logger.info(f"数据插入完成，共 {inserted} 条")
        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"数据插入失败：{e}")
            raise

    def fetch_questions(self):
        """只获取启用状态的问答对。"""
        try:
            self.cursor.execute(
                f"SELECT question, id FROM {self.table_name} WHERE status='enabled'"
            )
            result = self.cursor.fetchall()
            self.logger.info(f"成功获取 {len(result)} 条启用问题")
            return result
        except pymysql.MySQLError as e:
            self.logger.error(f"获取所有问题失败：{e}")
            return []

    def fetch_answer(self, question):
        try:
            self.cursor.execute(
                f"SELECT answer, id FROM {self.table_name} WHERE question=%s AND status='enabled'",
                (question,)
            )
            result = self.cursor.fetchone()
            if result:
                answer, qa_id = result
                self._increment_hit_count(qa_id)
                self.logger.info(f"问题命中并更新计数：{question}")
                return answer
            self.logger.info(f"问题未命中：{question}")
            return None
        except pymysql.MySQLError as e:
            self.logger.error(f"获取问题答案失败：{e}")
            return None

    def _increment_hit_count(self, qa_id):
        """增加问答对命中次数。"""
        try:
            self.cursor.execute(
                f"UPDATE {self.table_name} SET hit_count = hit_count + 1 WHERE id=%s",
                (qa_id,)
            )
            self.connection.commit()
        except Exception as e:
            self.logger.warning(f"更新命中计数失败：{e}")

    # ---- 管理 CRUD 接口 ----

    def add_qa(self, question, answer, domain="", created_by=""):
        """新增问答对。"""
        try:
            self.cursor.execute(
                f"INSERT INTO {self.table_name} (question, answer, domain, created_by) "
                "VALUES (%s, %s, %s, %s)",
                (question, answer, domain, created_by)
            )
            self.connection.commit()
            new_id = self.cursor.lastrowid
            self.logger.info(f"新增问答对 id={new_id}")
            return new_id
        except pymysql.MySQLError as e:
            self.connection.rollback()
            self.logger.error(f"新增问答对失败：{e}")
            return None

    def update_qa(self, qa_id, question=None, answer=None, domain=None, status=None):
        """更新问答对。"""
        fields = []
        values = []
        if question is not None:
            fields.append("question=%s")
            values.append(question)
        if answer is not None:
            fields.append("answer=%s")
            values.append(answer)
        if domain is not None:
            fields.append("domain=%s")
            values.append(domain)
        if status is not None:
            fields.append("status=%s")
            values.append(status)
        if not fields:
            return False
        values.append(qa_id)
        try:
            self.cursor.execute(
                f"UPDATE {self.table_name} SET {', '.join(fields)} WHERE id=%s",
                tuple(values)
            )
            self.connection.commit()
            self.logger.info(f"更新问答对 id={qa_id}")
            return True
        except pymysql.MySQLError as e:
            self.connection.rollback()
            self.logger.error(f"更新问答对失败：{e}")
            return False

    def delete_qa(self, qa_id, hard=False):
        """删除问答对。hard=True 物理删除，hard=False 软删除（设为 disabled）。"""
        if hard:
            query = f"DELETE FROM {self.table_name} WHERE id=%s"
        else:
            query = f"UPDATE {self.table_name} SET status='disabled' WHERE id=%s"
        try:
            self.cursor.execute(query, (qa_id,))
            self.connection.commit()
            self.logger.info(f"{'物理删除' if hard else '软删除'}问答对 id={qa_id}")
            return True
        except pymysql.MySQLError as e:
            self.connection.rollback()
            self.logger.error(f"删除问答对失败：{e}")
            return False

    def get_total_count(self):
        try:
            self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            return self.cursor.fetchone()[0]
        except pymysql.MySQLError as e:
            self.logger.error(f"统计失败：{e}")
            return 0

    def close(self):
        try:
            self.cursor.close()
            self.connection.close()
            self.logger.info('数据库连接关闭成功')
        except pymysql.MySQLError as e:
            self.logger.error(f"数据库连接关闭失败：{e}")
            raise


if __name__ == '__main__':
    client = LawMySQLClient()
    client.create_table()
    client.insert_data('law_qa_pairs.jsonl')
    print(f"表中总条数：{client.get_total_count()}")
    client.close()
