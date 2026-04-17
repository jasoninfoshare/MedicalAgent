"""
GraphDatabase/validators.py - Cypher 查询验证器
"""

import re
from typing import List, Tuple
from neo4j import GraphDatabase


class CypherValidator:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def validate_syntax(self, cypher_query: str) -> Tuple[bool, List[str]]:
        errors = []
        if not cypher_query.strip().upper().startswith(('MATCH', 'CREATE', 'MERGE', 'CALL')):
            errors.append("查询必须以MATCH, CREATE, MERGE 或 CALL开头!!!")
        if any(keyword in cypher_query.upper() for keyword in ['DROP', 'DELETE', 'DETACH', 'REMOVE']):
            if not any(keyword in cypher_query.upper() for keyword in ['DELETE', 'DETACH']):
                errors.append("查询包含可能危险的操作符")
        if cypher_query.upper().startswith('MATCH') and 'RETURN' not in cypher_query.upper():
            errors.append("MATCH查询必须包含RETURN语句!!!")
        try:
            with self.driver.session() as session:
                session.run(f"EXPLAIN {cypher_query}")
                return True, errors
        except Exception as e:
            errors.append(f"语法错误: {str(e)}")
            return False, errors

    def validate_against_schema(self, cypher_query: str, schema) -> Tuple[bool, List[str]]:
        errors = []
        node_labels = [node.label for node in schema.nodes]
        node_pattern = r'\(([a-zA-Z0-9_]+)?:?([a-zA-Z0-9_]+)\)'
        matches = re.findall(node_pattern, cypher_query)
        for match in matches:
            if match[1] and match[1] not in node_labels:
                errors.append(f"使用了不存在的节点标签: {match[1]}")

        rel_types = [rel.type for rel in schema.relationships]
        rel_pattern = r'\[([a-zA-Z0-9_]+)?:?([a-zA-Z0-9_]+)\]'
        rel_matches = re.findall(rel_pattern, cypher_query)
        for match in rel_matches:
            if match[1] and match[1] not in rel_types:
                errors.append(f"使用了不存在的关系类型: {match[1]}")

        return len(errors) == 0, errors

    def close(self):
        self.driver.close()


class RuleBasedValidator:
    def validate(self, cypher_query: str, schema) -> Tuple[bool, List[str]]:
        errors = []
        if not cypher_query.strip():
            errors.append("查询不能为空!!!")
            return False, errors

        dangerous_patterns = [
            (r'(?i)drop\s+', "DROP操作可能危险"),
            (r'(?i)delete\s+', "DELETE操作需要谨慎"),
            (r'(?i)detach\s+delete', "DETACH DELETE操作非常危险"),
            (r'(?i)remove\s+', "REMOVE操作需要谨慎"),
        ]
        for pattern, message in dangerous_patterns:
            if re.search(pattern, cypher_query):
                errors.append(message)

        if re.match(r'(?i)match', cypher_query) and not re.search(r'(?i)return', cypher_query):
            errors.append("MATCH查询必须包含RETURN子句")

        if re.match(r'(?i)create', cypher_query) and not re.search(r'(?i)(node|relationship|label|index)', cypher_query):
            errors.append("CREATE查询应该明确创建节点或关系")

        return len(errors) == 0, errors
