# TRIBE V2 API Template Architecture Notes

This document records the architecture decisions made so far for a clean,
API-focused TRIBE V2 hosting template. It is intentionally not an implementation
plan yet. A separate planning pass should break this into milestones,
checkpoints, tests, and deployment steps.

## Goal

Create a minimal, low-cost inference API for Meta TRIBE V2 that can be deployed
on a cheap CPU VPS and called by a separate frontend. The API should accept text
and static images, run pretrained TRIBE V2 inference asynchronously, and return
frontend-renderable brain activation outputs.

The first version should optimize for:

- Getting pretrained inference running reliably.
- Keeping hosting cost low.
- Reusing proven patterns from existing working repositories.
- Producing outputs that a separate UI can render later.
- Staying within non-commercial research/internal-use guardrails.

## Primary Deployment Target

V1 targets a cheap CPU VPS.

Assumptions:

- Ubuntu VPS.
- Docker-first if practical.
- CPU-only PyTorch wheels.
- No GPU requirement.
- One worker process.
- One job at a time.
- API only, not a hosted full web application.

RunPod, Vast.ai, Modal, and other burst/GPU platforms may be useful later, but
they are not v1 targets. The code should avoid unnecessary provider lock-in so a
future GPU worker path remains possible.

## Product Scope

### V1 Inputs

Supported:

- Text input.
- Static image input.

Not supported in v1:

- Video uploads.
- Audio uploads.
- Training or fine-tuning.
- Public monetized use.
- Experimental ad/brainrot scoring categories.

### Text Handling

Use the official TRIBE V2 inference pathway as much as possible, but avoid
WhisperX for v1 text.

Decision:

- Do not expose audio input.
- Avoid running WhisperX on CPU.
- Use the user's known text to create CPU-safe word timing events.
- Drop text/word events before prediction by default because the official text
  feature extractor uses gated Llama weights. Enable them only with
  `TRIBE_ENABLE_TEXT_EVENTS=true` and valid Hugging Face access.
- Reuse the `script-brain-optimizer` approach as reference: patch
  `ExtractWordsFromAudio._get_transcript_from_audio` so generated audio can be
  paired with heuristic word timings from the source text.

Important reference:

- `research-repos/hf-spaces/script-brain-optimizer/app.py`
  - `_patched_get_transcript_from_audio`
  - `_CURRENT_SCRIPT_TEXT`
  - `apply_patches`
  - text flow inside `analyze`

Notes:

- Upstream `TribeModel.get_events_dataframe(text_path=...)` converts text to
  speech with `gTTS`, then transcribes to word-level events. For CPU v1, avoid
  transcribing generated speech with WhisperX.
- The CPU-safe word timing approach is a pragmatic adapter, not a claim that
  TRIBE V2 has a native pure-text inference endpoint.

### Image Handling

Use the proven pattern from existing Spaces:

1. Accept JPEG, PNG, and WebP initially.
2. Normalize with Pillow to RGB PNG.
3. Use ffmpeg to encode a short silent H.264 MP4.
4. Call TRIBE V2 via `model.get_events_dataframe(video_path=...)`.

Decision:

- Treat image input as "static image as short visual stimulus."
- Use a default duration of 3 seconds.
- Use ffmpeg directly, not MoviePy.
- Use `yuv420p` for broad ffmpeg/browser compatibility.
- Preserve aspect ratio where possible; cap max dimension around 480-640 px.

Important references:

- `research-repos/hf-spaces/brainrot-or-not/app.py`
  - `image_to_video`
  - `scan_image`
- `research-repos/hf-spaces/ad-brain-scorer/app.py`
  - `_image_to_video`
  - `_get_events`
  - `analyze_single`

Upstream TRIBE V2 reference:

- `research-repos/tribev2/tribev2/eventstransforms.py`
  - `CreateVideosFromImages`
- `research-repos/tribev2/tribev2/demo_utils.py`
  - `TribeModel.get_events_dataframe`

Observation:

- Upstream public inference helper exposes `text_path`, `audio_path`, and
  `video_path`, not `image_path`.
- Existing working image demos adapt images into the supported video path.

## API Shape

V1 should be asynchronous.

Recommended shape:

- `POST /predict/text`
- `POST /predict/image`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result.json`
- `GET /jobs/{job_id}/preds.norm.f16.bin`
- `GET /metadata`
- `GET /health`

Auth:

- No bearer token in v1.
- The API should be private-by-network instead.
- Deployment docs should assume SSH tunnel, firewall, or private reverse proxy.
- Do not expose an unauthenticated public prediction endpoint.

Queue:

- Simple in-process queue.
- One job at a time.
- No Redis/RQ/Celery for v1.
- No need to preserve queued/running jobs across restarts.
- Completed and failed job folders can remain readable from disk until TTL.

## Job Storage

Each job should be a folder on disk.

Suggested layout:

```text
jobs/
  <job_id>/
    input/
    events.json
    result.json
    preds.raw.npy
    preds.norm.f16.bin
    logs.txt
    metadata.json
```

Retention:

- Delete successful uploaded inputs after processing by default.
- Keep result artifacts for 24 hours by default.
- Keep failed job logs for the same TTL.
- Failed inputs should only be kept if `TRIBE_KEEP_FAILED_INPUTS=true`.

Suggested env vars:

- `TRIBE_RESULT_TTL_HOURS=24`
- `TRIBE_DELETE_INPUTS_AFTER_JOB=true`
- `TRIBE_KEEP_FAILED_INPUTS=false`
- `TRIBE_MAX_TEXT_CHARS=5000`
- `TRIBE_MAX_IMAGE_MB=10`
- `TRIBE_IMAGE_STIMULUS_SECONDS=3`
- `TRIBE_FORCE_CPU=true`
- `TRIBE_ENABLE_TEXT_EVENTS=false`

## Output Contract

Return both reduced JSON outputs and optional full vertex render assets.

Default JSON should include:

- Job status.
- Input type.
- Timings and durations.
- Yeo7 network means.
- Heuristic axes.
- PCA components/variance if computed.
- Paths/URLs for binary prediction artifacts.
- Model and license metadata.

Full render asset:

- Normalized `float16` binary prediction blob.
- Shape: `[T, 20484]`.
- Intended for frontend brain rendering.

Internal debug artifact:

- Raw predictions as `.npy`.
- Not necessarily exposed as a public endpoint.

Reference for output reduction:

- `research-repos/auto-excitement/server.py`
  - `_reduce`
  - `_save_preds_binary`
  - `_dump_fsaverage5_mesh`

## Brain Mesh And Atlas Assets

Decision:

- The API should own and serve canonical render assets for v1.
- This avoids frontend/API mismatch in vertex ordering or atlas mapping.

Assets:

- fsaverage5 mesh binary for frontend rendering.
- Yeo7 atlas projection to fsaverage5.
- Prediction blob aligned to the same vertex ordering.

Important references:

- `research-repos/auto-excitement/build_atlas.py`
  - Builds `atlas_yeo7_fsaverage5.npz`.
- `research-repos/auto-excitement/server.py`
  - `_dump_fsaverage5_mesh`.

## Scoring And Categories

V1 should stay conservative.

Include:

- Yeo7 network summaries.
- PCA summaries.
- Heuristic axes from `auto-excitement`, clearly labeled as heuristic proxies:
  - `excitement`
  - `valence`
  - `cognitive_load`
  - `novelty`

Do not include in v1:

- Brainrot score.
- Ad effectiveness score.
- Emotional engagement score.
- Reward and motivation score.
- Social cognition score.
- Other marketing-style categories.

Future reference:

- `research-repos/hf-spaces/brainrot-or-not/app.py`
  - `SCORE_REGIONS`
  - `compute_scores`
- `research-repos/hf-spaces/ad-brain-scorer/brain_regions.py`
  - `REGION_GROUPS`
  - `compute_ad_scores`

Reason to defer:

- Those categories may be useful UI inspiration later, but they create stronger
  psychological and marketing claims than v1 should make.

## CPU Hardening

Decision:

- Include CPU hardening patches, gated by `TRIBE_FORCE_CPU=true`.
- Prefer clean config/device overrides first.
- Apply blunt monkey patches only where upstream or dependencies still try to
  use CUDA.

Important references:

- `research-repos/hf-spaces/ad-brain-scorer/app.py`
  - `CUDA_VISIBLE_DEVICES=""`
  - `_redirect_cuda_args`
  - patched `torch.nn.Module.to`
  - patched `torch.Tensor.to`
  - patched `torch.Tensor.cuda`
  - patched `torch.cuda.is_available`
- `research-repos/tribeV2_ViralAnalyser/tribe_runtime.py`
  - `_select_torch_device`
  - `_build_runtime_config_update`
  - `_prepare_runtime_model_dir`
  - `_drop_text_events_unless_enabled`

Important caveat:

- CPU-only inference may be slow. The accepted latency target is approximately
  10-20 minutes per input for v1.

## Docker Notes

Docker-first is tentative: use it if it behaves well, but avoid getting stuck if
TRIBE dependencies fight the container.

Reference base:

- `research-repos/hf-spaces/script-brain-optimizer/Dockerfile`

Useful packages seen in references:

- `python:3.11-slim`
- `git`
- `ffmpeg`
- `libsm6`
- `libxext6`
- `libgl1`
- `libsndfile1`

Reference CPU torch approach:

- `research-repos/hf-spaces/ad-brain-scorer/requirements.txt`
  - `--extra-index-url https://download.pytorch.org/whl/cpu`
  - `torch`
  - `torchvision`

Model weights:

- Download on first container start.
- Persist into mounted cache volume.
- Do not bake model weights into the Docker image.

## Metadata Endpoint

Include `/metadata`.

It should return:

- Wrapper/template version.
- TRIBE V2 model repo: `facebook/tribev2`.
- TRIBE source commit or pinned reference.
- Model license: `CC-BY-NC-4.0`.
- Mesh: `fsaverage5`.
- Vertex count: `20484`.
- Subject model: average subject.
- Hemodynamic lag: 5 seconds.
- Feature flags:
  - `force_cpu`
  - `text_word_timing_patch`
  - `image_as_video_adapter`
- Non-commercial/research disclaimer.

## License And Use Guardrails

TRIBE V2 is licensed under CC BY-NC 4.0.

V1 assumptions:

- Private, non-commercial research/internal use.
- No paid API access.
- No client-facing ad optimization product.
- No monetized output usage.
- No claims of measuring actual viewer brains.

Recommended language:

- Outputs are predicted fMRI-like responses from a computational model.
- Outputs are not direct brain measurements.
- Outputs are for non-commercial research/evaluation.
- The model predicts an average-subject response on fsaverage5.

References:

- `research-repos/tribev2/LICENSE`
- `research-repos/tribev2/README.md`
- `research-repos/tribeV2_ViralAnalyser/README.md`
- `research-repos/tribeV2_ViralAnalyser/NOTICE.md`

## Known Open Risks

- CPU inference may be much slower than hoped.
- TRIBE/neuralset may require CUDA hardening beyond simple `device="cpu"`.
- Docker image size may be large.
- First model load and dependency download may be slow.
- Text word timing patch is approximate.
- Image-as-video is an adapter, not native image inference.
- Exact frontend mesh rendering contract still needs implementation design.
- Mac local development may not match Linux VPS behavior, especially around
  PyTorch, ffmpeg, and system libraries.

## Next Planning Pass

The next document should be a real implementation plan with:

- Milestones.
- Checkpoints.
- Local Mac feasibility checks.
- Linux/Docker smoke tests.
- VPS setup steps.
- Test fixture strategy.
- API schema examples.
- Failure modes and recovery behavior.
- Minimal frontend integration contract.
