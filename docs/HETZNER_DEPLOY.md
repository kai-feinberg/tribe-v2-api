# Hetzner Text MVP Deployment

This is the first deployment target for the text-only MVP. Keep the service
private until an auth layer exists.

## VPS

The first proven deployment used:

- Ubuntu
- 2 vCPU
- 4 GB RAM
- 8 GB swap

This is enough for one text job at a time on CPU. If model loading or inference
fails with memory pressure, first add swap; then resize to 4 vCPU / 8 GB RAM
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

Health check:

```bash
curl http://127.0.0.1:8000/health
```

The default Compose file binds the API to `127.0.0.1` on the VPS, so it is
intended to be reached through an SSH tunnel rather than exposed publicly.

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

For the first Hetzner test server:

```bash
ssh -i ~/.ssh/hetzner_tribev2 -L 8000:localhost:8000 root@204.168.145.117
```
