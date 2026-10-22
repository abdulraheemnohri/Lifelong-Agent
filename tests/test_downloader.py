"""Downloader tests: path selection logic only (no network access)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lifelong_agent.llm import downloader


def _touch(path: Path, size: int = 2_000_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_find_gguf_files(tmp_path: Path):
    model_dir = tmp_path / "Youtu-LLM-2B-GGUF"
    _touch(model_dir / "model.Q4_K_M.gguf")
    _touch(model_dir / "notes.txt")
    _touch(model_dir / "sub" / "other.gguf", size=10)
    files = downloader.find_gguf_files(model_dir)
    names = {p.name for p in files}
    assert names == {"model.Q4_K_M.gguf", "other.gguf"}


def test_find_gguf_files_missing_dir(tmp_path: Path):
    assert downloader.find_gguf_files(tmp_path / "nope") == []


def test_select_best_gguf_prefers_q4_k_m(tmp_path: Path):
    files = [
        _touch(tmp_path / "model.Q8_0.gguf"),
        _touch(tmp_path / "model.Q4_K_M.gguf"),
        _touch(tmp_path / "model.Q4_0.gguf"),
    ]
    assert downloader.select_best_gguf(files).name == "model.Q4_K_M.gguf"


def test_select_best_gguf_prefers_q4_0_then_q5_k_m(tmp_path: Path):
    files = [
        _touch(tmp_path / "model.Q5_K_M.gguf"),
        _touch(tmp_path / "model.Q4_0.gguf"),
    ]
    assert downloader.select_best_gguf(files).name == "model.Q4_0.gguf"


def test_select_smallest_when_no_quant_match(tmp_path: Path):
    files = [
        _touch(tmp_path / "big.gguf", size=5_000_000),
        _touch(tmp_path / "small.gguf", size=2_000_000),
    ]
    assert downloader.select_best_gguf(files).name == "small.gguf"


def test_select_best_gguf_requires_files():
    with pytest.raises(ValueError):
        downloader.select_best_gguf([])


def test_exact_cli_command_shape():
    command = downloader._build_cli_command(
        downloader.DEFAULT_REPO, downloader.DEFAULT_REVISION, Path("Youtu-LLM-2B-GGUF")
    )
    assert command[:3] == ["huggingface-cli", "download", "tencent/Youtu-LLM-2B"]
    assert "--revision" in command and "GGUF" in command
    assert "--include" in command and "*.gguf" in command
    assert "--local-dir" in command and "Youtu-LLM-2B-GGUF" in command


def test_ensure_model_skips_download_when_present(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "model"
    existing = _touch(model_dir / "m.Q4_K_M.gguf")

    def _no_download(*args, **kwargs):
        raise AssertionError("download must not run when the model exists")

    monkeypatch.setattr(downloader, "_try_huggingface_cli", _no_download)
    monkeypatch.setattr(downloader, "_try_python_module", _no_download)
    monkeypatch.setattr(downloader, "_try_snapshot_download", _no_download)

    result = downloader.ensure_model(local_dir=model_dir)
    assert result == existing


def test_ensure_model_raises_without_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(downloader, "_try_huggingface_cli", lambda *a, **k: False)
    monkeypatch.setattr(downloader, "_try_python_module", lambda *a, **k: False)
    monkeypatch.setattr(downloader, "_try_snapshot_download", lambda *a, **k: False)
    with pytest.raises(RuntimeError):
        downloader.ensure_model(local_dir=tmp_path / "empty")
