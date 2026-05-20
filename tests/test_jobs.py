from dataclasses import replace

from tribev2_api.config import Settings
from tribev2_api.jobs import create_job, load_job


def test_create_and_load_text_job(tmp_path):
    settings = replace(Settings(), jobs_dir=tmp_path, cache_dir=tmp_path / "cache")

    record = create_job(settings, "text")
    loaded = load_job(settings, record.job_id)

    assert loaded is not None
    assert loaded.job_id.startswith("txt_")
    assert loaded.status == "queued"
    assert (tmp_path / loaded.job_id / "metadata.json").exists()
