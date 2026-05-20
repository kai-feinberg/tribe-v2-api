# TRIBE V2 API Template Implementation Plan

This plan turns the architecture notes into an execution path. The priority is
to reach a deployable text-only MVP as quickly as possible, prove it on a cheap
Hetzner CPU VPS, and only then add image support.

The key principle for v1 is:

> Do the smallest amount of work needed to submit text, run pretrained TRIBE V2
> on CPU, and retrieve frontend-renderable brain activation artifacts.

Image support, richer scoring, auth, object storage, GPU workers, and frontend
visualization are intentionally deferred until the text path is proven.

## Current Decisions

- Build a clean template repo.
- API only.
- CPU VPS first.
- Primary provider target: Hetzner Cloud.
- Baseline VPS target: 4 vCPU / 8 GB RAM, with 16 GB RAM noted as the safer
  fallback.
- Docker Compose is the primary deployment interface.
- Model/cache setup should be explicit, not hidden inside first API boot.
- Text inference comes before image inference.
- Raw JSON text input only for v1.
- No audio/video upload endpoints in v1.
- No generated temporary audio/video artifacts exposed by default.
- One running job at a time, with a small in-process pending queue.
- Local disk job/result storage.
- Readable job ID prefixes.
- Mac local development only needs light tests; Linux Docker/VPS is the real
  inference target.

## Success Criteria

The MVP is successful when all of the following are true:

- A fresh Hetzner Ubuntu VPS can clone the template.
- `docker compose build` succeeds.
- An explicit setup command downloads or prepares the model cache.
- The API starts with CPU-only settings.
- `POST /predict/text` accepts raw JSON text and creates a job.
- A text job completes on CPU without WhisperX.
- `GET /jobs/{job_id}` reports status and artifact URLs.
- `GET /jobs/{job_id}/result.json` returns reduced frontend-friendly output.
- `GET /jobs/{job_id}/preds.norm.f16.bin` returns the normalized vertex blob.
- `GET /metadata` returns model, license, mesh, feature, and wrapper metadata.
- A CLI smoke test can submit text and fetch outputs.

## Non-Goals For The MVP

- Native image inference.
- Image-as-video adapter.
- Audio input.
- Video input.
- Training or fine-tuning.
- Public hosted web UI.
- Authentication system.
- Redis, Celery, RQ, or a durable queue.
- S3/R2/object storage.
- GPU support.
- Marketing, ad effectiveness, or brainrot scoring categories.
- Claims that outputs measure actual viewer brains.

## Milestone 0: Repo Skeleton And Runtime Shape

Goal:

Create the minimal project structure needed to build, configure, run, test, and
deploy the API template.

Deliverables:

- `README.md`
- `ARCHITECTURE.md`
- `IMPLEMENTATION_PLAN.md`
- `docker-compose.yml`
- `Dockerfile`
- `.env.example`
- `Makefile`
- `requirements.txt` or equivalent dependency file.
- `src/tribev2_api/` package.
- `scripts/` folder for setup and smoke tests.
- `fixtures/` folder with one tiny text fixture.
- `jobs/.gitkeep`

Recommended structure:

```text
tribev2-api-template/
  src/tribev2_api/
    __init__.py
    app.py
    config.py
    metadata.py
    queue.py
    jobs.py
    inference/
      __init__.py
      cpu.py
      text.py
      outputs.py
      patches.py
  scripts/
    setup_model.py
    submit_text.py
    smoke_text.py
  fixtures/
    sample.txt
  jobs/
    .gitkeep
  Dockerfile
  docker-compose.yml
  Makefile
  .env.example
  README.md
  ARCHITECTURE.md
  IMPLEMENTATION_PLAN.md
```

Implementation notes:

- Keep source code small and boring.
- Avoid general abstractions until there is a real second implementation path.
- Use env vars documented in `ARCHITECTURE.md`.
- Mount model/cache and job directories as Docker volumes.
- Do not bake model weights into the image.

Checkpoint:

- `docker compose config` works.
- API can start and serve `/health` without loading TRIBE V2.
- Mac can run lightweight tests that do not import or load the full model.

## Milestone 1: CPU Runtime Probe

Goal:

Prove the runtime can import TRIBE V2 and load the pretrained model on CPU in a
Linux container before building the full API around it.

Deliverables:

- `scripts/setup_model.py`
- `scripts/probe_runtime.py`
- Docker command or Make target:

```bash
make setup-model
make probe-runtime
```

Probe behavior:

- Force CPU mode.
- Disable visible CUDA devices.
- Import PyTorch.
- Import TRIBE V2.
- Load `TribeModel.from_pretrained("facebook/tribev2", device="cpu")`.
- Print model/source metadata.
- Exit cleanly.

CPU hardening order:

1. Prefer explicit device/config settings.
2. Set `CUDA_VISIBLE_DEVICES=""`.
3. Use CPU-only PyTorch wheels.
4. Add monkey patches only if upstream dependencies still attempt CUDA access.

References:

- `research-repos/hf-spaces/ad-brain-scorer/app.py`
- `research-repos/tribeV2_ViralAnalyser/tribe_runtime.py`

Checkpoint:

- Model loads on the target Linux environment.
- Memory usage is recorded.
- Startup/load time is recorded.
- If 8 GB RAM fails, rerun on 16 GB before changing architecture.

Decision gate:

- If CPU model loading cannot be made reliable on Hetzner, pause and reassess
  before building Milestone 2.

## Milestone 2: Text Inference Core

Goal:

Run one text input through TRIBE V2 on CPU and produce raw predictions without
WhisperX.

Deliverables:

- `src/tribev2_api/inference/text.py`
- `src/tribev2_api/inference/patches.py`
- Small text inference script:

```bash
make infer-text-fixture
```

Text path:

1. Accept known source text.
2. Write it to a temporary text file.
3. Apply CPU-safe word timing patch.
4. Call the official TRIBE V2 text pathway as much as possible.
5. Capture events dataframe.
6. Run prediction.
7. Return predictions plus event/timing metadata.

Important constraint:

- Do not run WhisperX for v1 text.

Reference:

- `research-repos/hf-spaces/script-brain-optimizer/app.py`
  - `_patched_get_transcript_from_audio`
  - `_CURRENT_SCRIPT_TEXT`
  - `apply_patches`
  - `analyze`

Open implementation question to resolve in code:

- Whether to let upstream generate TTS audio and patch transcript timing, or
  bypass more of the audio/transcription path by directly constructing event
  data.

Preferred approach:

- Start with the proven `script-brain-optimizer` patch because it is known to
  work with upstream expectations.
- Only bypass deeper internals if TTS/network access or CPU latency becomes a
  blocker.

Checkpoint:

- A tiny fixture text produces a prediction array.
- Shape is confirmed and logged.
- Expected vertex count is confirmed as `20484`.
- Inference duration is measured.
- No CUDA dependency is required.

Decision gate:

- If text inference takes far beyond the accepted 10-20 minute range for small
  input, reduce default text length and document the limit before API work.

## Milestone 3: Minimal Async Text API

Goal:

Expose the working text inference path through a small API with an in-process
queue and disk-backed artifacts.

Deliverables:

- `src/tribev2_api/app.py`
- `src/tribev2_api/config.py`
- `src/tribev2_api/queue.py`
- `src/tribev2_api/jobs.py`
- `scripts/submit_text.py`

Endpoints:

```text
GET  /health
GET  /metadata
POST /predict/text
GET  /jobs/{job_id}
GET  /jobs/{job_id}/result.json
GET  /jobs/{job_id}/preds.norm.f16.bin
```

`POST /predict/text` request:

```json
{
  "text": "A short piece of text to evaluate."
}
```

Immediate response:

```json
{
  "job_id": "txt_20260516_abcd1234",
  "status": "queued",
  "status_url": "/jobs/txt_20260516_abcd1234"
}
```

Queue behavior:

- One active job.
- Up to 3 pending jobs by default.
- Reject additional jobs with `429 Too Many Requests`.
- Running and queued jobs do not need to survive restart.
- Completed and failed job folders survive restart if the jobs volume is
  mounted.

Job folder:

```text
jobs/
  txt_20260516_abcd1234/
    events.json
    result.json
    preds.raw.npy
    preds.norm.f16.bin
    logs.txt
    metadata.json
```

Checkpoint:

- Submit text.
- Poll job.
- Job completes.
- Result endpoints return expected artifacts.
- Failed jobs return useful errors and logs.

## Milestone 4: Output Contract And Metadata

Goal:

Make the output useful to a future frontend without building the frontend yet.

Deliverables:

- `src/tribev2_api/inference/outputs.py`
- `src/tribev2_api/metadata.py`
- Normalized binary output writer.
- Reduced JSON writer.

`result.json` should include:

- `job_id`
- `status`
- `input_type`
- `created_at`
- `started_at`
- `completed_at`
- `duration_seconds`
- `model`
- `license`
- `mesh`
- `prediction`
- `time`
- `network_traces`
- `heuristic_axes`
- `artifacts`

Time fields:

- Native timestep indices.
- Seconds.
- Input/event timing metadata where available.

Prediction binary:

- Normalized `float16`.
- Shape `[T, 20484]`.
- Served as `preds.norm.f16.bin`.
- Raw predictions stored internally as `preds.raw.npy`.

Reductions:

- Yeo7 network summaries.
- PCA summary if cheap enough.
- Heuristic axes from `auto-excitement`, clearly labeled as heuristic proxies.
- No ad/brainrot categories in v1.

References:

- `research-repos/auto-excitement/server.py`
  - `_reduce`
  - `_save_preds_binary`
  - `_dump_fsaverage5_mesh`
- `research-repos/auto-excitement/build_atlas.py`

`/metadata` should include:

- Wrapper version.
- TRIBE V2 model repo: `facebook/tribev2`.
- TRIBE source reference.
- License: `CC-BY-NC-4.0`.
- Mesh: `fsaverage5`.
- Vertex count: `20484`.
- Subject model: average subject.
- Hemodynamic lag: 5 seconds.
- Enabled feature flags.
- Non-commercial/research disclaimer.
- URLs for upstream project and license.

Checkpoint:

- `result.json` is stable enough for frontend integration.
- Binary artifact shape and dtype are documented.
- Metadata endpoint gives enough context for a future UI About panel.

## Milestone 5: Hetzner Text MVP Deployment

Goal:

Deploy the text-only API to a cheap Hetzner VPS and run real smoke tests there.

Deliverables:

- `docs/HETZNER_DEPLOY.md`
- VPS setup commands.
- Firewall/private-network guidance.
- SSH tunnel usage.
- Smoke test instructions.

Target setup:

- Ubuntu VPS.
- 4 vCPU / 8 GB RAM baseline.
- Docker and Docker Compose plugin.
- Persistent cache mount.
- Persistent jobs mount.
- API bound privately or firewalled.

Suggested directories:

```text
/opt/tribev2-api/
/opt/tribev2-cache/
/opt/tribev2-jobs/
```

Security posture:

- No bearer token in v1.
- Do not expose public unauthenticated inference.
- Use SSH tunnel, firewall allowlist, or private reverse proxy.

Smoke test:

1. `make setup-model`
2. `make probe-runtime`
3. `docker compose up -d`
4. `python scripts/submit_text.py --url http://localhost:8000 --text "..."`.
5. Poll until complete.
6. Download `result.json`.
7. Download `preds.norm.f16.bin`.

Checkpoint:

- A text job completes on Hetzner.
- Resource use is documented.
- Latency is documented.
- Any required VPS size adjustment is recorded.

Decision gate:

- Do not implement image support until text MVP has completed at least one
  successful VPS inference job.

## Milestone 6: Documentation For Text MVP

Goal:

Make the template usable from scratch by someone following the README.

Deliverables:

- Updated `README.md`.
- Environment variable reference.
- API examples.
- Troubleshooting guide.
- License/use guardrail section.

README flow:

1. Clone repo.
2. Copy `.env.example` to `.env`.
3. Start Docker.
4. Run explicit model setup.
5. Start API.
6. Submit text.
7. Fetch result artifacts.

Troubleshooting should cover:

- Model download failures.
- Out-of-memory errors.
- CUDA/device errors.
- ffmpeg missing, even though image is deferred.
- Slow CPU inference.
- Job failure logs.

Checkpoint:

- A fresh reader can understand the deployment path without reading source code.

## Milestone 7: Image Support After Text Is Proven

Goal:

Add static image input using the proven image-as-short-video adapter.

This milestone starts only after text MVP is working on the VPS.

Deliverables:

- `POST /predict/image`
- `src/tribev2_api/inference/image.py`
- Image fixture.
- Image smoke test script.

Image path:

1. Accept JPEG, PNG, and WebP.
2. Enforce max size.
3. Normalize with Pillow to RGB.
4. Preserve aspect ratio with padding to a fixed canvas.
5. Encode short silent H.264 MP4 with ffmpeg.
6. Call TRIBE V2 `video_path`.
7. Produce the same artifact contract as text jobs.

Default:

- 3 second image stimulus.

References:

- `research-repos/hf-spaces/brainrot-or-not/app.py`
  - `image_to_video`
  - `scan_image`
- `research-repos/hf-spaces/ad-brain-scorer/app.py`
  - `_image_to_video`
  - `_get_events`
  - `analyze_single`

Checkpoint:

- One image fixture completes on VPS.
- Result schema matches text result schema where possible.
- Temporary MP4 is deleted after successful processing.

## Milestone 8: Future Provider And GPU Options

Goal:

Document but do not implement alternate runtime paths.

Deferred options:

- RunPod serverless or pod.
- Vast.ai.
- Modal.
- GPU worker mode.
- Separate queue and worker process.
- Object storage for artifacts.
- Public API auth.

Reason to defer:

- CPU text MVP should establish whether the model and output contract are
  useful before adding provider complexity.

## Test Strategy

Mac local tests:

- Config parsing.
- Job folder creation.
- Queue behavior.
- API request validation.
- Result file serving with fake artifacts.
- Output normalization using small fake arrays.

Linux Docker tests:

- Import TRIBE V2.
- Load model on CPU.
- Run text fixture inference.
- Run API text job.
- Fetch artifacts.

VPS smoke tests:

- Full text job.
- Restart container and verify completed job artifacts remain readable.
- Submit too many jobs and verify queue rejection.
- Confirm no public access unless explicitly configured.

Manual acceptance tests:

- `GET /metadata` is truthful and complete.
- Failed jobs include enough log detail to debug.
- Latency and memory are captured in notes.

## Risk Register

CPU model loading fails:

- Try 16 GB RAM VPS.
- Confirm CPU-only PyTorch wheels.
- Apply CPU hardening patches.
- Reassess whether GPU/burst provider is required.

Text path depends on network TTS:

- Prefer proven patch first.
- If network/TTS is a blocker, construct events directly from known text.

WhisperX path accidentally runs:

- Patch transcript extraction.
- Add smoke test/log assertion.
- Keep audio input out of v1.

Output shape mismatch:

- Assert vertex count.
- Include shape in `metadata.json`.
- Fail loudly if shape is unexpected.

Docker image becomes too large:

- Keep weights in mounted cache.
- Avoid baking artifacts into image.
- Use slim base only if dependencies still work.

Mac/Linux drift:

- Keep Mac tests light.
- Treat Linux Docker/VPS as runtime truth.

License misuse:

- Include non-commercial metadata.
- Keep API private.
- Avoid monetized-output workflows.
- Avoid strong claims about real individual brain activity.

## Implementation Order

1. Build skeleton and health/metadata stubs.
2. Add Docker Compose and explicit setup commands.
3. Probe TRIBE CPU model load in Linux Docker.
4. Implement text fixture inference script.
5. Implement async text API.
6. Implement output artifacts.
7. Deploy text MVP on Hetzner.
8. Run and document smoke tests.
9. Only then start image support.

## Immediate Next Work Session

The next coding session should start with Milestone 0 and only enough of
Milestone 1 to answer this question:

> Can the container import TRIBE V2 and load the pretrained model on CPU?

If the answer is yes, continue toward text inference. If the answer is no, fix
runtime/dependency issues before building more API surface.
