"""CPU Gradio demo for the from-scratch NumPy and PyTorch TinySAM models."""

import argparse
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinysam.config import NUMPY_TINY, TORCH_SMALL
from tinysam.data import IMAGENET_MEAN, IMAGENET_STD
from tinysam.numpy_impl.model import TinySAM as NumpyTinySAM
from tinysam.torch_impl.convert import load_numpy_weights, load_torch_weights
from tinysam.torch_impl.model import TinySAM as TorchTinySAM


class TinySAMApp:
    """Load one backend and reuse its image embedding for repeated prompt clicks."""

    def __init__(self, impl="numpy", weights=None, device="cpu"):
        self.impl, self.device = impl, device
        self.config = NUMPY_TINY if impl == "numpy" else TORCH_SMALL
        default_name = "tinysam_numpy.npz" if impl == "numpy" else "tinysam_torch.pt"
        self.weights = Path(weights or os.environ.get(
            f"TINYSAM_{impl.upper()}_WEIGHTS", ROOT / "app" / "weights" / default_name
        ))
        if impl == "numpy":
            self.model = NumpyTinySAM(self.config)
            if self.weights.exists():
                load_numpy_weights(self.model, self.weights)
            else:
                raise FileNotFoundError(f"NumPy demo weights not found: {self.weights}")
        else:
            import torch
            self.model = TorchTinySAM(self.config).to(device)
            if not self.weights.exists():
                raise FileNotFoundError(f"PyTorch demo weights not found: {self.weights}")
            load_torch_weights(self.model, self.weights, map_location=device)
            self.model.eval()
        self.cached_key = None
        self.cached_embedding = None

    def predict(self, image, points, labels):
        """Return an original-resolution RGB overlay and the best predicted IoU."""
        image = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.uint8)
        key = hashlib.sha1(image.tobytes()).hexdigest()
        resized = np.asarray(Image.fromarray(image).resize(
            (self.config.img_size, self.config.img_size), Image.Resampling.BILINEAR
        ), dtype=np.float32).transpose(2, 0, 1)
        normalized = (resized / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        model_points = np.zeros((1, 3, 2), dtype=np.float32)
        model_labels = np.full((1, 3), -1, dtype=np.int64)
        count = min(3, len(points), len(labels))
        if count:
            model_points[0, :count, 0] = points[:count, 0] * self.config.img_size / image.shape[1]
            model_points[0, :count, 1] = points[:count, 1] * self.config.img_size / image.shape[0]
            model_labels[0, :count] = labels[:count]
        if key != self.cached_key:
            if self.impl == "numpy":
                self.cached_embedding = self.model.encode_image(normalized[None])
            else:
                import torch
                with torch.no_grad():
                    self.cached_embedding = self.model.encode_image(torch.from_numpy(normalized[None]).to(self.device))
            self.cached_key = key
        if self.impl == "numpy":
            logits, quality = self.model.decode(self.cached_embedding, model_points, model_labels)
            best = int(quality[0].argmax())
            low = logits[0, best]
            low_image = Image.fromarray(low.astype(np.float32), mode="F").resize(
                (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
            )
            mask = np.asarray(low_image) > 0
            score = float(quality[0, best])
        else:
            import torch
            import torch.nn.functional as functional
            with torch.no_grad():
                logits, quality = self.model.decode(
                    self.cached_embedding,
                    torch.from_numpy(model_points).to(self.device),
                    torch.from_numpy(model_labels).to(self.device),
                )
                best = int(quality[0].argmax())
                resized_logits = functional.interpolate(
                    logits[:, best:best + 1], size=image.shape[:2], mode="bilinear", align_corners=False
                )[0, 0].cpu().numpy()
                mask = resized_logits > 0
                score = float(quality[0, best])
        overlay = image.astype(np.float32).copy()
        tint = np.asarray([45, 230, 115], dtype=np.float32)
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * tint
        return np.clip(overlay, 0, 255).astype(np.uint8), score


def make_app(default_impl="numpy"):
    import gradio as gr

    engines = {}

    def engine(impl, weights):
        key = (impl, weights or "")
        if key not in engines:
            engines[key] = TinySAMApp(impl, weights or None)
        return engines[key]

    with gr.Blocks(title="TinySAM from scratch") as demo:
        gr.Markdown("# TinySAM from scratch\nChoose an implementation, upload an image, then click prompts.")
        with gr.Row():
            impl = gr.Dropdown(["numpy", "torch"], value=default_impl, label="Implementation")
            weights = gr.Textbox(value="", label="Weights path (optional)")
            mode = gr.Radio(["positive", "negative", "box"], value="positive", label="Prompt mode")
        with gr.Row():
            image = gr.Image(type="numpy", label="Input image", interactive=True)
            overlay = gr.Image(type="numpy", label="Predicted mask overlay")
        quality = gr.Textbox(label="Predicted IoU")
        state = gr.State({"points": [], "labels": [], "image": None, "box_clicks": 0, "mode": None})
        clear = gr.Button("Clear prompts")

        def reset(image_value):
            state_value = {"points": [], "labels": [], "image": image_value, "box_clicks": 0, "mode": None}
            return image_value, "", state_value

        def on_select(image_value, impl_value, weight_value, mode_value, state_value, event: gr.SelectData):
            if image_value is None:
                return None, "Upload an image first.", state_value
            x, y = event.index[:2]
            points = list(state_value.get("points", []))
            labels = list(state_value.get("labels", []))
            if state_value.get("mode") not in (None, mode_value):
                points, labels = [], []
                state_value["box_clicks"] = 0
            if mode_value == "box":
                if len(points) >= 2 or state_value.get("box_clicks", 0) >= 2:
                    points, labels = [], []
                points.append([float(x), float(y)])
                labels.append(2 if len(points) == 1 else 3)
                state_value["box_clicks"] = len(points)
            else:
                points.append([float(x), float(y)])
                labels.append(1 if mode_value == "positive" else 0)
                points, labels = points[-3:], labels[-3:]
            state_value.update(points=points, labels=labels, image=image_value, mode=mode_value)
            try:
                prediction, score = engine(impl_value, weight_value).predict(image_value, points, labels)
                return prediction, f"{score:.3f}", state_value
            except FileNotFoundError as exc:
                return image_value, str(exc), state_value

        image.select(on_select, inputs=[image, impl, weights, mode, state],
                     outputs=[overlay, quality, state])
        clear.click(reset, inputs=[image], outputs=[overlay, quality, state])
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    args, _unknown = parser.parse_known_args()
    if args.weights:
        os.environ[f"TINYSAM_{args.impl.upper()}_WEIGHTS"] = args.weights
    make_app(default_impl=args.impl).launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
