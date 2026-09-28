import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny


def block(inp, out, stride=1):
    return nn.Sequential(nn.Conv2d(inp, out, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, out), nn.GELU())


class DroneSegV2(nn.Module):
    """Trainable RGB backbone + shallow derived-image details + global text FiLM."""
    def __init__(self, checkpoint=None, detail=True, text=True, boundary=True, refine=False, num_classes=5):
        super().__init__()
        backbone = convnext_tiny(weights=None)
        if checkpoint:
            backbone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.backbone = backbone.features
        self.use_detail, self.use_text, self.use_boundary = detail, text, boundary
        self.use_refine = refine
        if refine and not (detail and boundary):
            raise ValueError("The refinement head requires detail inputs and a boundary head")
        width = 128
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in [96, 192, 384, 768])
        self.smooth = nn.ModuleList(block(width, width) for _ in range(4))
        if detail:
            self.detail_stem = nn.Sequential(block(2, 32, 2), block(32, 64, 2))
            self.detail_down = block(64, 128, 2)
            self.detail_proj = nn.ModuleList([nn.Conv2d(64, width, 1), nn.Conv2d(128, width, 1)])
            self.detail_gate = nn.ModuleList([nn.Conv2d(width * 2, width, 1) for _ in range(2)])
            for gate in self.detail_gate:
                nn.init.constant_(gate.bias, -2)
        if text:
            self.text_film = nn.Sequential(nn.Linear(512, width), nn.GELU(), nn.Linear(width, width * 2))
            nn.init.zeros_(self.text_film[-1].weight)
            nn.init.zeros_(self.text_film[-1].bias)
        self.fuse = nn.Sequential(block(width * 4, width), nn.Dropout2d(0.1))
        self.classifier = nn.Conv2d(width, num_classes, 1)
        if boundary:
            self.edge_head = nn.Sequential(block(width, 32), nn.Conv2d(32, 1, 1))
        if refine:
            # Fuse half-resolution RGB/detail evidence with coarse semantic context.
            # Append after the original modules to keep their seeded initialization.
            self.refine_stem = nn.Sequential(block(5, 32, 2), block(32, 32))
            self.refine_context = nn.Conv2d(width, 32, 1)
            self.refine_body = nn.Sequential(block(65 + num_classes, 32), block(32, 32))
            self.refine_delta = nn.Conv2d(32, num_classes, 1)
            nn.init.zeros_(self.refine_delta.weight)
            nn.init.zeros_(self.refine_delta.bias)

    def forward(self, batch):
        image = batch["image"]
        x, levels = image, []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in (1, 3, 5, 7):
                levels.append(x)
        levels = [proj(x) for proj, x in zip(self.lateral, levels)]
        if self.use_text:
            gamma, beta = self.text_film(F.normalize(batch["text"].float(), dim=-1)).chunk(2, dim=1)
            levels[-1] = levels[-1] * (1 + gamma[..., None, None]) + beta[..., None, None]
        if self.use_detail:
            d1 = self.detail_stem(batch["detail"])
            for i, d in enumerate([d1, self.detail_down(d1)]):
                d = self.detail_proj[i](d)
                gate = torch.sigmoid(self.detail_gate[i](torch.cat([levels[i], d], dim=1)))
                levels[i] = levels[i] + gate * d
        for i in (2, 1, 0):
            levels[i] = levels[i] + F.interpolate(levels[i+1], size=levels[i].shape[-2:], mode="bilinear", align_corners=False)
        levels = [conv(x) for conv, x in zip(self.smooth, levels)]
        fused = self.fuse(torch.cat([F.interpolate(x, size=levels[0].shape[-2:], mode="bilinear", align_corners=False) for x in levels], dim=1))
        coarse = self.classifier(fused)
        result = {"logits": F.interpolate(coarse, size=image.shape[-2:], mode="bilinear", align_corners=False)}
        if self.use_boundary:
            edge = self.edge_head(fused)
            result["boundary"] = F.interpolate(edge, size=image.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_refine:
            low = self.refine_stem(torch.cat([image, batch["detail"]], dim=1))
            size = low.shape[-2:]
            context = F.interpolate(self.refine_context(fused), size=size, mode="bilinear", align_corners=False)
            coarse_half = F.interpolate(coarse, size=size, mode="bilinear", align_corners=False)
            edge_half = F.interpolate(edge, size=size, mode="bilinear", align_corners=False).sigmoid()
            correction = self.refine_delta(self.refine_body(torch.cat([low, context, coarse_half, edge_half], dim=1)))
            # A nonzero floor permits correcting interior errors as well as borders.
            correction = correction * (0.25 + 0.75 * edge_half)
            result["coarse_logits"] = result["logits"]
            result["logits"] = result["coarse_logits"] + F.interpolate(correction, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return result
