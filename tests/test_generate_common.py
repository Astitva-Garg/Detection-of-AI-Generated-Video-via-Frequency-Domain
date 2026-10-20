import json

import numpy as np

from src.generate.common import Job, prompt_list, run_jobs, t2v_jobs


def _jobs(n=3):
    return [Job(f"g_{i}", "g", "p", "s", i, f"prompt_{i}") for i in range(n)]


def _ok(job):
    return np.zeros((8, 32, 32, 3), np.uint8), 8


def test_run_jobs_writes_and_resumes(tmp_path):
    calls = []

    def gen(job):
        calls.append(job.id)
        return _ok(job)

    run_jobs(_jobs(), gen, tmp_path)
    assert len(list(tmp_path.glob("*.mp4"))) == 3
    assert len((tmp_path / "manifest.jsonl").read_text().splitlines()) == 3
    run_jobs(_jobs(), gen, tmp_path)
    assert len(calls) == 3


def test_run_jobs_failure_is_logged_and_retried(tmp_path):
    state = {"fail": True}

    def gen(job):
        if job.id == "g_1" and state["fail"]:
            raise RuntimeError("CUDA out of memory")
        return _ok(job)

    run_jobs(_jobs(), gen, tmp_path)
    recs = [json.loads(l) for l in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert [r["status"] for r in recs] == ["ok", "failed", "ok"]
    state["fail"] = False
    run_jobs(_jobs(), gen, tmp_path)
    assert (tmp_path / "g_1.mp4").exists()


def test_t2v_jobs_groups():
    n = len(prompt_list())
    jobs = t2v_jobs("ltx", n + 2)
    assert jobs[0].group_id == jobs[n].group_id
    assert len({j.id for j in jobs}) == n + 2
