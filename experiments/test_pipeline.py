import numpy as np
import pytest
import torch

from experiments.data import decode_mask, COLORS, DroneData, DATA_ROOT
from experiments.losses import SoftDice, edge_bce, Confusion
from experiments.inference import sliding_logits, positions
from utils import FocalLoss, EdgeLoss


def test_palette_and_unknown_color():
    assert decode_mask(COLORS[None]).tolist() == [[0, 1, 2, 3, 4]]
    with pytest.raises(ValueError, match="Unknown label colors"):
        decode_mask(np.zeros((1, 1, 3), np.uint8))


@pytest.mark.parametrize("kind", ["dice", "focal", "legacy_edge", "boundary"])
def test_loss_backpropagates_with_ignore(kind):
    logits = torch.randn(2, 5, 12, 16, requires_grad=True)
    target = torch.randint(0, 5, (2, 12, 16))
    target[:, :2] = 255
    if kind == "dice":
        loss = SoftDice()(logits, target)
    elif kind == "focal":
        loss = FocalLoss()(logits, target)
    elif kind == "legacy_edge":
        loss = EdgeLoss()(logits, target)
    else:
        loss = edge_bce(logits[:, :1], target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert logits.grad[:, :, :2].abs().sum() == 0


def test_all_ignored_losses_are_zero():
    logits = torch.randn(2, 5, 8, 8, requires_grad=True)
    target = torch.full((2, 8, 8), 255, dtype=torch.long)
    for loss in [SoftDice()(logits, target), FocalLoss()(logits, target), edge_bce(logits[:, :1], target)]:
        assert torch.isfinite(loss) and loss.item() == 0


def test_streaming_metrics_known_example():
    m = Confusion(2)
    m.update(torch.tensor([0, 0]), torch.tensor([0, 1]))
    m.update(torch.tensor([1, 1, 0]), torch.tensor([1, 0, 255]))
    result = m.compute()
    assert result["miou"] == pytest.approx(1/3)
    assert result["f1"] == pytest.approx(0.5)
    assert result["pixel_accuracy"] == pytest.approx(0.5)


def test_sliding_reconstructs_logits_and_covers_borders():
    class PixelModel(torch.nn.Module):
        def forward(self, batch):
            x = batch["image"][:, :1]
            return {"logits": torch.cat([x + i for i in range(5)], 1)}
    batch = {"image": torch.randn(1, 3, 43, 71)}
    model = PixelModel()
    expected = model(batch)["logits"]
    actual = sliding_logits(model, batch, crop=32, stride=24, amp=False)
    assert torch.allclose(expected, actual, atol=1e-6)
    small = {"image": torch.randn(1, 3, 7, 9)}
    assert torch.allclose(sliding_logits(model, small, 32, 24, False), model(small)["logits"])


def test_real_data_reproducible_and_aligned():
    val = DroneData(DATA_ROOT, "val", text=True, detail=True, limit=1)
    a, b = val[0], val[0]
    for key in ["image", "mask", "text", "detail"]:
        assert torch.equal(a[key], b[key])
    train = DroneData(DATA_ROOT, "train", crop=128, text=True, detail=True, limit=1)
    train.epoch = 3
    a, b = train[0], train[0]
    assert torch.equal(a["image"], b["image"])
    assert set(a["mask"].unique().tolist()) <= {0, 1, 2, 3, 4, 255}
    train.epoch = 4
    assert not torch.equal(a["image"], train[0]["image"])
