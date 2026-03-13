---
title: "Deploying AI Applications: From Local Dev to Production"
description: "Step-by-step guide to deploying AI applications — FastAPI serving, Docker containers, cloud deployment on AWS/GCP, model caching, rate limiting, and monitoring."
date: "2026-03-10"
slug: "deploying-ai-applications"
keywords: ["deploying AI applications", "AI app deployment", "FastAPI AI", "AI production deployment"]
---

## Learning Objectives

- Build a production-ready AI API with FastAPI
- Containerize with Docker
- Deploy to cloud (AWS, GCP, or Railway)
- Implement caching, rate limiting, and error handling
- Monitor AI app performance and costs

---

## Architecture Overview

```
Client
  ↓  HTTPS
Load Balancer / CDN
  ↓
FastAPI Server (multiple replicas)
  ↓              ↓              ↓
LLM API    Vector DB      Cache (Redis)
(OpenAI)   (Qdrant)       (Redis)
                ↓
          Monitoring
        (Prometheus/Grafana)
```

---

## Step 1: Build the API with FastAPI

```bash
pip install fastapi uvicorn pydantic openai python-dotenv
```

```python
# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, RateLimitError
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Assistant API", version="1.0.0")

# CORS — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    max_tokens: int = 512


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    tokens_used: int


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if len(req.message) > 4000:
        raise HTTPException(status_code=400, detail="Message too long (max 4000 chars)")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user",   "content": req.message},
            ],
            max_tokens=req.max_tokens,
        )
        return ChatResponse(
            answer=response.choices[0].message.content,
            conversation_id=req.conversation_id or "new",
            tokens_used=response.usage.total_tokens,
        )
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

### Streaming Endpoint

```python
from fastapi.responses import StreamingResponse
import asyncio

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": req.message}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield f"data: {chunk.choices[0].delta.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## Step 2: Add Caching with Redis

Cache identical requests to save money and reduce latency.

```bash
pip install redis
docker run -d -p 6379:6379 redis:alpine
```

```python
import redis
import hashlib
import json

cache = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)
CACHE_TTL = 3600  # 1 hour

def get_cache_key(message: str) -> str:
    return f"chat:{hashlib.sha256(message.encode()).hexdigest()}"

@app.post("/chat", response_model=ChatResponse)
async def chat_with_cache(req: ChatRequest):
    cache_key = get_cache_key(req.message)

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        data = json.loads(cached)
        return ChatResponse(**data, tokens_used=0)

    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": req.message}],
        max_tokens=req.max_tokens,
    )

    result = ChatResponse(
        answer=response.choices[0].message.content,
        conversation_id=req.conversation_id or "new",
        tokens_used=response.usage.total_tokens,
    )

    # Cache the result
    cache.setex(cache_key, CACHE_TTL, json.dumps(result.dict()))
    return result
```

---

## Step 3: Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Too many requests"})

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")  # 20 requests per minute per IP
async def chat(request: Request, req: ChatRequest):
    ...
```

---

## Step 4: Dockerize

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Non-root user for security
RUN adduser --disabled-password --gecos '' appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_HOST=redis
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

```bash
docker-compose up -d
curl http://localhost:8000/health
```

---

## Step 5: Cloud Deployment

### Railway (Easiest — 5 minutes)

```bash
npm install -g @railway/cli
railway login
railway init
railway up
railway domain  # get your public URL
```

Set environment variables in Railway dashboard.

### AWS ECS (Container Service)

```bash
# Build and push to ECR
aws ecr create-repository --repository-name ai-api
aws ecr get-login-password | docker login --username AWS --password-stdin <ecr-url>
docker build -t ai-api .
docker tag ai-api:latest <ecr-url>/ai-api:latest
docker push <ecr-url>/ai-api:latest

# Deploy with Terraform or AWS Console
```

### Google Cloud Run (Serverless)

```bash
gcloud run deploy ai-api \
  --image gcr.io/your-project/ai-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=sk-...
```

---

## Step 6: Monitoring

### Track Token Usage and Costs

```python
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class UsageLog:
    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float

COST_PER_1K_TOKENS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o":      {"input": 0.0025,  "output": 0.01},
}

def log_usage(response, model: str):
    usage = response.usage
    costs = COST_PER_1K_TOKENS.get(model, {"input": 0, "output": 0})
    cost = (
        usage.prompt_tokens / 1000 * costs["input"]
        + usage.completion_tokens / 1000 * costs["output"]
    )
    log = UsageLog(
        timestamp=datetime.utcnow().isoformat(),
        model=model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=round(cost, 6),
    )
    logger.info(f"USAGE: {log}")
    return log
```

### Health Checks and Alerting

```python
@app.get("/health/detailed")
async def detailed_health():
    checks = {}

    # Check OpenAI connectivity
    try:
        client.models.list()
        checks["openai"] = "healthy"
    except Exception as e:
        checks["openai"] = f"error: {e}"

    # Check Redis
    try:
        cache.ping()
        checks["redis"] = "healthy"
    except Exception:
        checks["redis"] = "unreachable"

    status = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}
```

---

## Troubleshooting

**API returns 500 intermittently**
- Add retry logic with exponential backoff for OpenAI calls
- Check logs for specific error codes (rate limits, context length exceeded)

**High latency (> 5s)**
- Profile: is latency in LLM call or elsewhere?
- Use streaming for perceived responsiveness
- Cache frequent queries

**Costs are higher than expected**
- Log all token usage
- Check for prompt inflation (system prompt too long)
- Consider gpt-4o-mini for simpler tasks

---

## FAQ

**Should I use FastAPI or Flask?**
FastAPI: async support, automatic OpenAPI docs, better performance. Flask: simpler for tiny APIs. For AI apps with streaming, use FastAPI.

**How do I handle API key security?**
Never hard-code API keys. Use environment variables, AWS Secrets Manager, or HashiCorp Vault. Rotate keys regularly. Set spending limits in your OpenAI account.

---

## What to Learn Next

- **AI application architecture** → ai-application-architecture
- **Building AI chatbots** → building-ai-chatbots
- **LangChain in production** → [LangChain Complete Tutorial](/blog/langchain-tutorial-complete/)
