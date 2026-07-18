# Deploying to IBM Cloud Code Engine

This project deploys as a single container: FastAPI serves both the JSON
API (`/api/*`) and the static dashboard (`/`) from one process, which
matches Code Engine's "one container, one port" model cleanly — no need
for separate frontend/backend services or a load balancer.

## Prerequisites

- IBM Cloud CLI installed (`ibmcloud`) and logged in: `ibmcloud login --sso`
- Code Engine plugin: `ibmcloud plugin install code-engine`
- Container Registry plugin (or use Code Engine's build-from-source instead
  of a registry — see Option B below): `ibmcloud plugin install container-registry`
- Your watsonx.ai API key + project ID (used as env vars, never baked into the image)

## Option A — build locally, push to IBM Container Registry, deploy

```bash
# 1. Target a resource group / region
ibmcloud target -g <your-resource-group> -r <region, e.g. us-south>

# 2. Create (once) a Code Engine project
ibmcloud ce project create --name insider-threat-nlp

# 3. Create a namespace in IBM Container Registry (once)
ibmcloud cr namespace-add insider-threat-nlp-ns

# 4. Build and push the image
docker build -t us.icr.io/insider-threat-nlp-ns/insider-threat-nlp:latest -f deployment/Dockerfile .
ibmcloud cr login
docker push us.icr.io/insider-threat-nlp-ns/insider-threat-nlp:latest

# 5. Create the Code Engine app, wiring secrets from your .env values
ibmcloud ce secret create --name watsonx-secrets \
  --from-literal WATSONX_API_KEY=<your-key> \
  --from-literal WATSONX_PROJECT_ID=<your-project-id> \
  --from-literal WATSONX_URL=https://us-south.ml.cloud.ibm.com \
  --from-literal ANONYMIZATION_SALT=<a-long-random-string>

ibmcloud ce application create \
  --name insider-threat-nlp \
  --image us.icr.io/insider-threat-nlp-ns/insider-threat-nlp:latest \
  --registry-secret <your-registry-secret-if-private> \
  --env-from-secret watsonx-secrets \
  --cpu 1 --memory 2G \
  --min-scale 0 --max-scale 3 \
  --port 8000

# 6. Get the public URL
ibmcloud ce application get --name insider-threat-nlp --output url
```

## Option B — build from source directly on Code Engine (no local Docker needed)

```bash
ibmcloud ce project create --name insider-threat-nlp
ibmcloud ce project select --name insider-threat-nlp

ibmcloud ce application create \
  --name insider-threat-nlp \
  --build-source . \
  --build-dockerfile deployment/Dockerfile \
  --env-from-secret watsonx-secrets \
  --cpu 1 --memory 2G \
  --min-scale 0 --max-scale 3 \
  --port 8000
```

This has Code Engine build the image server-side from your local project
directory — simplest path if you don't want to manage a registry yourself.

## Updating after a code change

```bash
ibmcloud ce application update --name insider-threat-nlp --build-source .
# or, if using Option A's registry flow: rebuild, push, then:
ibmcloud ce application update --name insider-threat-nlp \
  --image us.icr.io/insider-threat-nlp-ns/insider-threat-nlp:latest
```

## Notes

- `--min-scale 0` lets the app scale to zero when idle (cost-efficient for
  an internship demo); the first request after idle will have a cold-start
  delay while the container boots and NLTK/sklearn import.
- The container runs `python -m src.pipeline` once at startup using
  whatever's in `data/raw/` at build time — for a live demo with the real
  CERT dataset, either bake the CSV into the image (COPY it in before
  `docker build`) or mount it from IBM Cloud Object Storage and adjust
  `src/config.py`'s `DATA_RAW` path accordingly.
- `ANONYMIZATION_SALT` must be set via secret, not committed — if it
  changes, previously-generated pseudonyms won't match new ones.
