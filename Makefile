.PHONY: compose-config build up down setup-model probe-runtime infer-text-fixture smoke-text test

compose-config:
	docker compose config

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

setup-model:
	docker compose run --rm api python scripts/setup_model.py

probe-runtime:
	docker compose run --rm api python scripts/probe_runtime.py

infer-text-fixture:
	docker compose run --rm api python scripts/infer_text_fixture.py --text-file fixtures/sample.txt

smoke-text:
	python scripts/smoke_text.py --url http://localhost:$${TRIBE_PORT:-8000} --text-file fixtures/sample.txt

test:
	PYTHONPATH=src TRIBE_FAKE_INFERENCE=true pytest -q
