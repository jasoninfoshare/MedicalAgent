"""
GraphDatabase/main.py - Neo4j NL2Cypher FastAPI 服务
启动方式：python GraphDatabase/main.py
服务端口：http://0.0.0.0:8101
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from openai import OpenAI
from dotenv import load_dotenv

from models import NL2CypherRequest, CypherResponse, ValidationRequest, ValidationResponse
from schemas import EXAMPLE_SCHEMA
from prompts import create_system_prompt, create_validation_prompt
from validators import CypherValidator, RuleBasedValidator

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if all([neo4j_uri, neo4j_user, neo4j_password]):
        app.state.validator = CypherValidator(neo4j_uri, neo4j_user, neo4j_password)
    else:
        app.state.validator = RuleBasedValidator()
    yield
    if hasattr(app.state.validator, 'close'):
        app.state.validator.close()


app = FastAPI(title="NL2Cypher API", lifespan=lifespan)

client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_cypher_query(natural_language: str, query_type: str = None) -> str:
    system_prompt = create_system_prompt(str(EXAMPLE_SCHEMA.model_dump()))
    user_prompt = natural_language
    if query_type:
        user_prompt = f"{query_type}查询: {natural_language}"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2048,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DeepSeek API错误: {str(e)}")


def explain_cypher_query(cypher_query: str) -> str:
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个Neo4j专家,请用简单明了的语言解释Cypher查询."},
                {"role": "user", "content": f"请解释以下Cypher查询:\n{cypher_query}"}
            ],
            temperature=0.1,
            max_tokens=1024,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"无法生成解释: {str(e)}"


@app.post("/generate", response_model=CypherResponse)
async def generate_cypher(request: NL2CypherRequest):
    cypher_query = generate_cypher_query(
        request.natural_language_query,
        request.query_type.value if request.query_type else None
    )
    explanation = explain_cypher_query(cypher_query)
    is_valid, errors = app.state.validator.validate_against_schema(cypher_query, EXAMPLE_SCHEMA)
    confidence = 0.9
    if errors:
        confidence = max(0.3, confidence - len(errors) * 0.1)
    return CypherResponse(
        cypher_query=cypher_query,
        explanation=explanation,
        confidence=confidence,
        validated=is_valid,
        validation_errors=errors
    )


@app.post("/validate", response_model=ValidationResponse)
async def validate_cypher(request: ValidationRequest):
    is_valid, errors = app.state.validator.validate_against_schema(request.cypher_query, EXAMPLE_SCHEMA)
    suggestions = []
    if errors:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个Neo4j专家,请提供Cypher查询的改进建议."},
                    {"role": "user", "content": create_validation_prompt(request.cypher_query)}
                ]
            )
            suggestions = [response.choices[0].message.content.strip()]
        except Exception:
            suggestions = ["无法生成建议"]
    return ValidationResponse(is_valid=is_valid, errors=errors, suggestions=suggestions)


@app.get("/schema")
async def get_schema():
    return EXAMPLE_SCHEMA.model_dump()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8101)
