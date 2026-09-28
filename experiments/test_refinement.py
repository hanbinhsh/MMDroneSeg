import pytest
import torch
from torch.nn import functional as F

from experiments.losses import hard_pixel_ce
from experiments.models import build_model, batch_needs
from experiments.train import make_optimizer
from experiments.evaluate_ours_tta import four_flip_logits


def test_hard_pixels_exclude_ignored_labels_and_backpropagate():
    logits = torch.tensor([[[[4., -4., 100.]], [[-4., 4., -100.]]]], requires_grad=True)
    target = torch.tensor([[[0, 0, 255]]])
    loss = hard_pixel_ce(logits, target, fraction=0.5)
    expected = F.cross_entropy(logits[..., 1:2], target[..., 1:2])
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad[..., 1].abs().sum() > 0
    assert logits.grad[..., 0].abs().sum() == 0
    assert logits.grad[..., 2].abs().sum() == 0
    assert hard_pixel_ce(logits, torch.full_like(target, 255)).item() == 0


def test_refinement_preserves_initial_output_and_can_learn():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    original = build_model("ours_sched50", pretrained=False).eval()
    torch.manual_seed(42)
    refined = build_model("ours_refine", pretrained=False).eval()
    assert batch_needs("ours_refine") == {"detail": True, "text": False}
    for key, value in original.state_dict().items():
        assert torch.equal(value, refined.state_dict()[key])
    batch = {"image": torch.randn(1, 3, 128, 128), "detail": torch.randn(1, 2, 128, 128)}
    with torch.no_grad():
        old = original(batch)["logits"]
    output = refined(batch)
    assert torch.equal(old, output["logits"])
    assert output["boundary"].shape == (1, 1, 128, 128)
    target = torch.randint(5, (1, 128, 128))
    F.cross_entropy(output["logits"], target).backward()
    grad = refined.refine_delta.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_short_budget_scheduler_reaches_zero_at_50():
    model = torch.nn.Linear(2, 2)
    optimizer, scheduler = make_optimizer(model, "ours_sched50", 50)
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[-1]["lr"] == pytest.approx(0)


def test_flip_ensemble_inverts_spatial_transforms():
    class PixelModel(torch.nn.Module):
        def forward(self, batch):
            value = batch["image"][:, :1] + batch["detail"][:, :1]
            return {"logits": torch.cat([value+i for i in range(5)], dim=1)}
    model = PixelModel()
    batch = {"image": torch.randn(1, 3, 19, 29), "detail": torch.randn(1, 2, 19, 29)}
    original, averaged = four_flip_logits(model, batch, crop=16, amp=False)
    assert torch.allclose(original, model(batch)["logits"], atol=1e-6)
    assert torch.allclose(averaged, original, atol=1e-6)
