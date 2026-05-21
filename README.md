# TRIBE V2 API Template

Minimal API-only template for running pretrained Meta TRIBE V2 inference on a
CPU VPS. The first target is text-only inference; image support is deferred
until the text path is proven on the VPS.

## Quick Start

```bash
cp .env.example .env
docker compose build
make setup-model
make probe-runtime
docker compose up -d
python scripts/smoke_text.py --url http://localhost:8000 --text-file fixtures/sample.txt
```

On a Mac, use `make test` for light tests. Real TRIBE inference should be
validated in Linux Docker or on the target VPS.

## API

- `GET /health`
- `GET /metadata`
- `POST /predict/text`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result.json`
- `GET /jobs/{job_id}/preds.norm.f16.bin`

Text request:

```json
{
  "text": "A short piece of text to evaluate."
}
```

## Browser Frontends

Set `TRIBE_CORS_ORIGINS` to the comma-separated browser origins that should be
allowed to call the API, for example:

```bash
TRIBE_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,https://your-frontend.vercel.app
```

## License And Use

TRIBE V2 is licensed under CC BY-NC 4.0. This template is intended for private
non-commercial research/internal evaluation. Outputs are predicted fMRI-like
responses from a computational model, not measurements of an individual brain.
