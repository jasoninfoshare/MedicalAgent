"""
GraphDatabase/schemas.py - Neo4j 图数据库 Schema 定义
"""

from pydantic import BaseModel
from typing import Dict, List


class NodeSchema(BaseModel):
    label: str
    properties: Dict[str, str]


class RelationshipSchema(BaseModel):
    type: str
    from_node: str
    to_node: str
    properties: Dict[str, str]


class GraphSchema(BaseModel):
    nodes: List[NodeSchema]
    relationships: List[RelationshipSchema]


EXAMPLE_SCHEMA = GraphSchema(
    nodes=[
        NodeSchema(label="Disease", properties={"name": "string"}),
        NodeSchema(label="Drug", properties={"name": "string"}),
        NodeSchema(label="Food", properties={"name": "string"}),
        NodeSchema(label="Symptom", properties={"name": "string"})
    ],
    relationships=[
        RelationshipSchema(
            type="has_symptom",
            from_node="Disease",
            to_node="Symptom",
            properties={}
        ),
        RelationshipSchema(
            type="recommand_drug",
            from_node="Disease",
            to_node="Drug",
            properties={}
        ),
        RelationshipSchema(
            type="recommand_eat",
            from_node="Disease",
            to_node="Food",
            properties={}
        ),
    ]
)


if __name__ == '__main__':
    res = str(EXAMPLE_SCHEMA.model_dump())
    print(res)
