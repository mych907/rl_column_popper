import json


def test_run_manifest_roundtrip_and_files(tmp_path):
    from column_popper.utils.manifest import RunManifest, write_json, append_jsonl

    m = RunManifest(env_id="SpecKitAI/ColumnPopper-v1", seed=42, reward_preset="default", meta={"k": 1})
    s = m.to_json()
    m2 = RunManifest.from_json(s)
    assert m2.env_id == m.env_id
    assert m2.seed == m.seed
    assert m2.reward_preset == m.reward_preset
    assert m2.meta == m.meta

    # File IO helpers
    jf = tmp_path / "manifest.json"
    write_json(str(jf), m)
    jf_text = jf.read_text().strip()
    obj = json.loads(jf_text)
    assert obj["env_id"] == "SpecKitAI/ColumnPopper-v1"

    jl = tmp_path / "frames.jsonl"
    append_jsonl(str(jl), {"a": 1})
    append_jsonl(str(jl), {"b": 2})
    lines = jl.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["a"] == 1
    assert json.loads(lines[1])["b"] == 2

