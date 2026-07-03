# Deployment

The app is packaged as a single Docker image that bundles the code, the prompt
config, the **prebuilt** vector index (`chroma_db/`), and the embedding +
reranker models (baked in at build time). The only runtime input is the
`OPENAI_API_KEY`, injected as an environment variable by the host.

```
docker build -t sec-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... sec-rag
# open http://localhost:8000
```

---

## Two things that make ML deploys different (and how we handle them)

**1. The prebuilt index has to reach the platform.**
`chroma_db/` (~176MB) is gitignored on purpose — a large, regenerable binary
doesn't belong in git (GitHub rejects files >100MB). So we don't deploy "from the
repo"; we build the image **locally (or in CI)** where `chroma_db/` exists, then
push that image to a registry and have the platform run it.

**2. Memory.** torch + bge-small + the cross-encoder reranker need **~1GB RAM** at
runtime. That rules out Render's *free* 512MB tier (it OOMs on model load). Pick a
target with ≥1GB.

| Platform | Free RAM | Fits? | Notes |
|---|---|---|---|
| **Hugging Face Spaces** (Docker) | ~16GB | ✅ best free option | public URL, ideal for ML demos |
| **Fly.io** | 256MB free, 2GB configurable | ⚠️ needs a paid-ish machine | good Docker support |
| **Render** | 512MB free / 2GB Standard | ❌ free / ✅ paid | matches "deployed on Render" story if you pay |

---

## Recommended free path — Hugging Face Spaces (Docker SDK)

1. Create a new **Space** → SDK: **Docker**.
2. Push this repo to the Space, tracking the index with git-lfs:
   ```bash
   git lfs install
   git lfs track "chroma_db/**"
   git add .gitattributes Dockerfile .dockerignore chroma_db
   git commit -m "Deploy: Dockerized app + prebuilt index"
   git push        # to the Space's git remote
   ```
3. In the Space **Settings → Variables and secrets**, add `OPENAI_API_KEY`.
4. The Space builds the Dockerfile and serves it at a public URL.

## Alternative — prebuilt image + Render/Fly

Build once where the index exists, push to a registry, point the platform at it:
```bash
docker build -t ghcr.io/<you>/sec-rag:latest .
docker push ghcr.io/<you>/sec-rag:latest
```
Then create a web service from that image, set `OPENAI_API_KEY`, and give it
≥1GB RAM.

---

## How this scales on AWS (the "I understand production" story)

The exact same image runs unchanged on:
- **ECS Fargate** — serverless containers; set the task memory to 2GB, put an
  Application Load Balancer in front, autoscale on CPU/requests.
- **EKS** — the image as a Deployment + Service behind an ingress.

Nothing about the app changes — that's the point of containerizing. For higher
scale you'd move the vector store off-box (managed Qdrant / pgvector) so replicas
share one index instead of each baking their own, and pull the models from a
shared volume or model registry.
