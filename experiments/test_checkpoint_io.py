import json

import pytest
import torch

from experiments.train import atomic_json, atomic_save, should_save_checkpoint


def test_checkpoint_rotation_retains_previous_complete_state(tmp_path):
    path = tmp_path / "last.pt"
    for epoch in (1, 2, 3):
        atomic_save(path, {"epoch": epoch, "model": torch.tensor([epoch])})
    assert torch.load(path, weights_only=True)["epoch"] == 3
    assert torch.load(tmp_path / "last.prev.pt", weights_only=True)["epoch"] == 2
    assert not (tmp_path / "last.pt.tmp").exists()


def test_failed_serialization_preserves_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    atomic_save(path, {"epoch": 1})

    def failed_save(data, stream):
        stream.write(b"incomplete checkpoint")
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(torch, "save", failed_save)
    with pytest.raises(OSError):
        atomic_save(path, {"epoch": 2})
    assert torch.load(path, weights_only=True)["epoch"] == 1


def test_atomic_json_writes_valid_utf8(tmp_path):
    path = tmp_path / "status.json"
    atomic_json(path, {"epoch": 5, "message": "已保存"})
    assert json.loads(path.read_text(encoding="utf-8"))["epoch"] == 5


def test_checkpoint_interval_and_final_or_stop_boundaries():
    saved = [epoch for epoch in range(1, 13) if should_save_checkpoint(epoch, 12, 5)]
    assert saved == [5, 10, 12]
    assert should_save_checkpoint(7, 50, 5, stopping=True)
    assert not should_save_checkpoint(7, 50, 5)
