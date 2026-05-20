# Hetzner Text MVP Deployment

This is the first deployment target for the text-only MVP. Keep the service
private until an auth layer exists.

## VPS

Start with:

- Ubuntu
- 4 vCPU
- 8 GB RAM

If model loading or inference fails with memory pressure, retry on 16 GB RAM
before changing the architecture.

## Server Setup

```bash
sudo apt-get update
sudo apt-get install -y git ca-certificates curl
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
```

Log out and back in so the Docker group takes effect.

## App Setup

```bash
sudo mkdir -p /opt/tribev2-api /opt/tribev2-cache /opt/tribev2-jobs
sudo chown -R "$USER":"$USER" /opt/tribev2-api /opt/tribev2-cache /opt/tribev2-jobs
cd /opt/tribev2-api
git clone <your-template-repo-url> .
cp .env.example .env
```

Edit `.env`:

```text
TRIBE_CACHE_HOST_DIR=/opt/tribev2-cache
TRIBE_JOBS_HOST_DIR=/opt/tribev2-jobs
TRIBE_FORCE_CPU=true
```

## First Runtime Check

```bash
docker compose build
make setup-model
make probe-runtime
```

Do not continue to API testing until `make probe-runtime` loads the model on
CPU successfully.

## Run API

```bash
docker compose up -d
docker compose logs -f api
```

In another shell:

```bash
python3 scripts/smoke_text.py --url http://localhost:8000 --text-file fixtures/sample.txt
```

## Private Access

V1 has no bearer token. Do not expose the prediction endpoint publicly.

Recommended access:

```bash
ssh -L 8000:localhost:8000 root@YOUR_SERVER_IP
```

Then call `http://localhost:8000` from your laptop.
