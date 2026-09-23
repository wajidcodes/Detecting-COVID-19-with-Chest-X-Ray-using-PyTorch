"""
Grad-CAM for ResNet-18 (last conv block = layer4).

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization." ICCV 2017. DOI: 10.1109/ICCV.2017.74
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fh = target_layer.register_forward_hook(self._fwd_hook)
        self._bh = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out) -> None:
        self._activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        with torch.enable_grad():
            x = image_tensor.clone().requires_grad_(True)
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            score = logits[0, class_idx]
            self.model.zero_grad()
            score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy()

    def remove_hooks(self) -> None:
        self._fh.remove()
        self._bh.remove()


def overlay_cam_on_image(original: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib.cm as cm

    img_w, img_h = original.size
    cam_uint8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    cam_pil = Image.fromarray(cam_uint8).resize((img_w, img_h), Image.BILINEAR)
    cam_float = np.array(cam_pil) / 255.0
    heat_rgb = (cm.jet(cam_float)[:, :, :3] * 255).astype(np.uint8)
    return Image.blend(original.convert("RGB"), Image.fromarray(heat_rgb), alpha)


def zone_scores(cam: np.ndarray) -> dict[str, float]:
    """
    Map a PA chest X-ray Grad-CAM onto six lung zones (Brixia-style layout).

    Image left = patient right lung. Rows go upper → lower.

    Borghesi & Maroldi, Radiol Med 2020; 125:509–513.
    DOI: 10.1007/s11547-020-01200-3

    This is a visualization of 2D attention, not a 3D CT segmentation.
    """
    h, w = cam.shape
    mid_x = w // 2
    y1, y2 = h // 3, 2 * h // 3
    regions = {
        "R_upper": cam[0:y1, 0:mid_x],
        "R_mid": cam[y1:y2, 0:mid_x],
        "R_lower": cam[y2:h, 0:mid_x],
        "L_upper": cam[0:y1, mid_x:w],
        "L_mid": cam[y1:y2, mid_x:w],
        "L_lower": cam[y2:h, mid_x:w],
    }
    return {k: round(float(v.mean()), 4) for k, v in regions.items()}


def image_to_b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    import base64

    return base64.b64encode(buf.getvalue()).decode("ascii")
