import torch
import torch.nn.functional as F


def positions(length, crop, stride):
    if length <= crop:
        return [0]
    return sorted(set(list(range(0, length - crop + 1, stride)) + [length - crop]))


@torch.no_grad()
def sliding_logits(model, batch, crop=512, stride=384, amp=True):
    """Accumulate logits, not argmax labels. No ground truth enters the model."""
    h, w = batch["image"].shape[-2:]
    spatial = {k: F.pad(batch[k], (0, max(0, crop-w), 0, max(0, crop-h)), mode="replicate")
               for k in ("image", "detail") if k in batch}
    hp, wp = spatial["image"].shape[-2:]
    logits = None
    counts = torch.zeros((1, 1, hp, wp), device=batch["image"].device)
    for y in positions(hp, crop, stride):
        for x in positions(wp, crop, stride):
            tile = {k: v[..., y:y+crop, x:x+crop] for k, v in spatial.items()}
            if "text" in batch:
                tile["text"] = batch["text"]
            with torch.autocast(device_type=counts.device.type, enabled=amp and counts.is_cuda):
                pred = model(tile)["logits"]
            if logits is None:
                logits = torch.zeros((pred.shape[0], pred.shape[1], hp, wp), device=pred.device)
            if pred.shape[-2:] != (crop, crop):
                pred = F.interpolate(pred.float(), size=(crop, crop), mode="bilinear", align_corners=False)
            logits[..., y:y+crop, x:x+crop] += pred.float()
            counts[..., y:y+crop, x:x+crop] += 1
    return (logits / counts.clamp_min(1))[..., :h, :w]
