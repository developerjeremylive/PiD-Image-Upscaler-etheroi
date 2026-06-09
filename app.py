import os
import sys
import subprocess
import tempfile
from typing import Iterable

import torch
import numpy as np
import gradio as gr
from PIL import Image
from types import SimpleNamespace
from huggingface_hub import snapshot_download

import spaces

DARK = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#FBE5C7", c100="#F5D29C", c200="#EFC174", c300="#E9B05A",
        c400="#E5A75B", c500="#E0A458", c600="#C68D3F", c700="#A6722E",
        c800="#7E5722", c900="#583C18", c950="#3A2810",
    ),
    neutral_hue=gr.themes.Color(
        c50="#E6E8EB", c100="#C9CDD3", c200="#ACB1B9", c300="#9097A0",
        c400="#7C8693", c500="#626972", c600="#4A4F58", c700="#363B43",
        c800="#262C35", c900="#1A1F26", c950="#12161B",
    ),
    font=(gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"),
).set(
    # Backgrounds
    body_background_fill="#12161B",
    background_fill_primary="#12161B",
    background_fill_secondary="#1A1F26",
    block_background_fill="#1A1F26",
    # Text
    body_text_color="#E6E8EB",
    body_text_color_subdued="#7C8693",
    # Borders
    border_color_primary="#262C35",
    border_color_accent="#E0A458",
    # Buttons — primary
    button_primary_background_fill="#E0A458",
    button_primary_background_fill_hover="#F0B870",
    button_primary_text_color="#12161B",
    button_primary_border_color="#E0A458",
    button_primary_border_color_hover="#F0B870",
    # Buttons — secondary
    button_secondary_background_fill="#1A1F26",
    button_secondary_background_fill_hover="#232930",
    button_secondary_text_color="#E6E8EB",
    button_secondary_border_color="#262C35",
    # Inputs
    input_background_fill="#12161B",
    input_border_color="#262C35",
    input_border_color_focus="#E0A458",
    input_border_width="1px",
    # Block styling
    block_border_width="1px",
    block_border_color="#262C35",
    block_label_background_fill="transparent",
    block_label_border_color="transparent",
    block_shadow="none",
    # Misc
    error_background_fill="#3A1E20",
    error_text_color="#F4A6A8",
    slider_color="#E0A458",
    # Checkbox
    checkbox_background_color="#12161B",
    checkbox_background_color_selected="#E0A458",
    checkbox_border_color="#262C35",
    checkbox_border_color_selected="#E0A458",
    checkbox_border_color_focus="#E0A458",
    checkbox_label_background_fill="#1A1F26",
    checkbox_label_background_fill_selected="rgba(224,164,88,0.10)",
    # Table
    table_border_color="#262C35",
    table_even_background_fill="#1A1F26",
    table_odd_background_fill="#12161B",
    # Panel
    panel_background_fill="#1A1F26",
    panel_border_color="#262C35",
    panel_border_width="1px",
)

css = """
/* ── Reset ALL radius ── */
*, *::before, *::after {
    border-radius: 0 !important;
}

/* ── Container ── */
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    background: #12161B !important;
}

/* ── Tab strip ── */
.tab-nav {
    border-bottom: 1px solid #262C35 !important;
    background: #12161B !important;
    padding: 0 !important;
}
.tab-nav button {
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    color: #7C8693 !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 10px 20px !important;
    background: transparent !important;
    transition: color 0.15s, border-color 0.15s, background 0.15s !important;
    margin-bottom: -1px !important;
}
.tab-nav button.selected,
.tab-nav button:hover {
    color: #E0A458 !important;
    border-bottom-color: #E0A458 !important;
    background: #1A1F26 !important;
}

/* ── Blocks / panels ── */
.gr-block, .gr-box, .gr-form,
div[data-testid="block"],
.block, .panel, .form,
.contain, .gap {
    border-radius: 0 !important;
}

/* ── Labels ── */
label span, .gr-block label span,
.block > label > span {
    color: #9097A0 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
}

/* ── Slider ── */
input[type=range] {
    accent-color: #E0A458 !important;
    border-radius: 0 !important;
}
input[type=range]::-webkit-slider-runnable-track {
    background: #262C35 !important;
    border-radius: 0 !important;
    height: 4px !important;
}
input[type=range]::-webkit-slider-thumb {
    background: #E0A458 !important;
    border-radius: 0 !important;
    width: 14px !important;
    height: 14px !important;
    margin-top: -5px !important;
    -webkit-appearance: none !important;
    appearance: none !important;
    border: none !important;
}
input[type=range]::-moz-range-track {
    background: #262C35 !important;
    border-radius: 0 !important;
    height: 4px !important;
}
input[type=range]::-moz-range-thumb {
    background: #E0A458 !important;
    border-radius: 0 !important;
    width: 14px !important;
    height: 14px !important;
    border: none !important;
}

/* ── All text inputs & textareas ── */
input[type=text],
input[type=number],
input[type=email],
input[type=search],
textarea,
select {
    background: #12161B !important;
    border: 1px solid #262C35 !important;
    color: #E6E8EB !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    outline: none !important;
}
input[type=text]:focus,
input[type=number]:focus,
input[type=email]:focus,
textarea:focus,
select:focus {
    border-color: #E0A458 !important;
    box-shadow: 0 0 0 1px #E0A458 !important;
}

/* ── Buttons ── */
button,
.gr-button {
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: background 0.15s, border-color 0.15s !important;
}
button.primary,
.gr-button-primary,
button[data-variant="primary"] {
    background: #E0A458 !important;
    color: #12161B !important;
    border: 1px solid #E0A458 !important;
    border-radius: 0 !important;
}
button.primary:hover,
.gr-button-primary:hover,
button[data-variant="primary"]:hover {
    background: #F0B870 !important;
    border-color: #F0B870 !important;
}
button.secondary,
.gr-button-secondary,
button[data-variant="secondary"] {
    background: #1A1F26 !important;
    color: #E6E8EB !important;
    border: 1px solid #262C35 !important;
    border-radius: 0 !important;
}
button.secondary:hover,
.gr-button-secondary:hover,
button[data-variant="secondary"]:hover {
    background: #232930 !important;
    border-color: #E0A458 !important;
}

/* ── Accordion / details ── */
details,
.gr-accordion {
    border: 1px solid #262C35 !important;
    border-radius: 0 !important;
    background: #1A1F26 !important;
    margin: 6px 0 !important;
}
details summary,
.gr-accordion summary {
    background: #1A1F26 !important;
    border-bottom: 1px solid transparent !important;
    border-radius: 0 !important;
    color: #9097A0 !important;
    padding: 9px 14px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    list-style: none !important;
    transition: color 0.15s, border-color 0.15s !important;
}
details summary:hover,
.gr-accordion summary:hover {
    color: #E0A458 !important;
    border-bottom-color: #E0A458 !important;
}
details[open] > summary {
    border-bottom: 1px solid #262C35 !important;
    color: #E0A458 !important;
}

/* ── Markdown ── */
.gr-markdown h1, .prose h1 {
    color: #E6E8EB !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    border-bottom: 1px solid #262C35 !important;
    padding-bottom: 8px !important;
    margin-bottom: 14px !important;
}
.gr-markdown h2, .prose h2 {
    color: #E6E8EB !important;
    font-size: 17px !important;
    font-weight: 600 !important;
}
.gr-markdown h3, .prose h3 {
    color: #C9CDD3 !important;
    font-size: 14px !important;
}
.gr-markdown p, .prose p {
    color: #9097A0 !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
.gr-markdown a, .prose a {
    color: #E0A458 !important;
    text-decoration: underline !important;
}
.gr-markdown strong, .prose strong {
    color: #E6E8EB !important;
}
.gr-markdown code, .prose code {
    background: #12161B !important;
    color: #E0A458 !important;
    border: 1px solid #262C35 !important;
    border-radius: 0 !important;
    padding: 1px 5px !important;
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Image upload zones ── */
.gr-image,
.upload-container,
[data-testid="image"],
.image-container,
[data-testid="image"] > div {
    background: #12161B !important;
    border: 1px solid #262C35 !important;
    border-radius: 0 !important;
}
.gr-image:hover,
.upload-container:hover {
    border-color: #E0A458 !important;
}

/* ── Image slider ── */
.image-slider-container,
[data-testid="imageslider"],
[data-testid="imageslider"] > div {
    border: 1px solid #262C35 !important;
    border-radius: 0 !important;
    background: #12161B !important;
}

/* ── Checkboxes ── */
input[type=checkbox] {
    accent-color: #E0A458 !important;
    border-radius: 0 !important;
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 14px !important;
    height: 14px !important;
    border: 1px solid #262C35 !important;
    background: #12161B !important;
    vertical-align: middle !important;
    position: relative !important;
    cursor: pointer !important;
    flex-shrink: 0 !important;
}
input[type=checkbox]:checked {
    background: #E0A458 !important;
    border-color: #E0A458 !important;
}
input[type=checkbox]:checked::after {
    content: "✓" !important;
    position: absolute !important;
    top: -2px !important;
    left: 1px !important;
    font-size: 11px !important;
    color: #12161B !important;
    font-weight: 700 !important;
}
input[type=checkbox]:focus {
    box-shadow: 0 0 0 1px #E0A458 !important;
    outline: none !important;
}

/* ── Radio buttons ── */
input[type=radio] {
    accent-color: #E0A458 !important;
}

/* ── Progress bar ── */
.progress-bar,
[data-testid="progress"],
.progress-level,
.progress-level-inner {
    border-radius: 0 !important;
}
.progress-bar > div {
    background: #E0A458 !important;
    border-radius: 0 !important;
}

/* ── Dropdown / select menus ── */
ul[role=listbox],
[role=option],
.gr-dropdown {
    border: 1px solid #262C35 !important;
    border-radius: 0 !important;
    background: #1A1F26 !important;
    color: #E6E8EB !important;
}
[role=option]:hover {
    background: #232930 !important;
    color: #E0A458 !important;
}

/* ── Status / error cards ── */
.status-card {
    padding: 12px 16px !important;
    border-radius: 0 !important;
    background: #1A1F26 !important;
    border: 1px solid #262C35 !important;
}
.status-error {
    background: #3A1E20 !important;
    border-color: #F4A6A8 !important;
    color: #F4A6A8 !important;
}

/* ── PiD header ── */
.pid-header {
    padding: 18px 0 12px 0 !important;
    border-bottom: 2px solid #E0A458 !important;
    margin-bottom: 16px !important;
}
.pid-header h1 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #E6E8EB !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: -0.01em !important;
}
.pid-header h1 .accent {
    color: #E0A458 !important;
}
.pid-tipbar {
    padding: 8px 14px !important;
    background: #1A1F26 !important;
    border: 1px solid #262C35 !important;
    border-left: 3px solid #E0A458 !important;
    border-radius: 0 !important;
    font-size: 12px !important;
    color: #7C8693 !important;
    margin-top: 10px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    line-height: 1.6 !important;
}
.pid-tipbar a {
    color: #E0A458 !important;
    text-decoration: underline !important;
}
.pid-tipbar strong {
    color: #C9CDD3 !important;
    font-weight: 500 !important;
}

/* ────────────────────────────────────────────────────────────
   Dim info badge — SINGLE outer border only
   Gradio wraps gr.Markdown in several divs:
     .pid-dim-badge          ← our elem_classes hook (outer)
       div                   ← Gradio block wrapper
         div.prose / .gr-markdown  ← actual prose node
   We kill every border/shadow/background on every inner layer,
   then re-apply ONE clean border on the outermost wrapper.
   ──────────────────────────────────────────────────────────── */

/* 1. Strip borders from ALL descendant wrappers inside the badge */
.pid-dim-badge *,
.pid-dim-badge > div,
.pid-dim-badge > div > div,
.pid-dim-badge .block,
.pid-dim-badge .wrap,
.pid-dim-badge .prose,
.pid-dim-badge .gr-markdown,
.pid-dim-badge [class*="svelte"] {
    border: none !important;
    border-top: none !important;
    border-bottom: none !important;
    border-left: none !important;
    border-right: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    outline: none !important;
}

/* 2. Strip borders from the Gradio block container Gradio injects */
.pid-dim-badge.block,
.pid-dim-badge > .block,
div.pid-dim-badge {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* 3. The ONE real border lives here — outermost wrapper via the class selector */
div[class*="pid-dim-badge"],
.pid-dim-badge {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11.5px !important;
    color: #7C8693 !important;
    background: #12161B !important;
    border: 1px solid #262C35 !important;
    border-left: 3px solid #626972 !important;
    border-radius: 0 !important;
    padding: 6px 10px !important;
    margin: 4px 0 8px 0 !important;
    line-height: 1.5 !important;
    /* Ensure it paints as a block */
    display: block !important;
}

/* 4. Typography pass-through for the inner <p> */
.pid-dim-badge p,
.pid-dim-badge .prose p,
.pid-dim-badge .gr-markdown p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11.5px !important;
    color: #7C8693 !important;
    line-height: 1.5 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #12161B; border-radius: 0; }
::-webkit-scrollbar-thumb { background: #262C35; border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: #E0A458; }

/* ── HR dividers ── */
hr {
    border: none !important;
    border-top: 1px solid #262C35 !important;
    margin: 12px 0 !important;
}

/* ── Number input spinners ── */
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {
    opacity: 1 !important;
    background: #1A1F26 !important;
    border-left: 1px solid #262C35 !important;
}

/* ── File / upload button ── */
.file-upload button,
[data-testid="upload-btn"],
label[data-testid="upload-btn"] {
    border: 1px dashed #262C35 !important;
    border-radius: 0 !important;
    background: #12161B !important;
    color: #7C8693 !important;
    transition: border-color 0.15s, color 0.15s !important;
}
.file-upload button:hover,
[data-testid="upload-btn"]:hover {
    border-color: #E0A458 !important;
    color: #E0A458 !important;
}

/* ── Tooltip / popover ── */
.tippy-box,
[data-tippy-content] {
    border: 1px solid #262C35 !important;
    border-radius: 0 !important;
    background: #1A1F26 !important;
    color: #E6E8EB !important;
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Wrap svelte-generated wrappers ── */
.svelte-1gfkn6j,
.wrap,
.wrap.default,
.wrap.svelte-1gfkn6j {
    border-radius: 0 !important;
}
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PID_REPO_URL = "https://github.com/nv-tlabs/PiD.git"
PID_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PiD")

if not os.path.exists(PID_REPO_DIR):
    print(f"[pid] cloning {PID_REPO_URL} -> {PID_REPO_DIR}", flush=True)
    subprocess.check_call(["git", "clone", "--depth", "1", PID_REPO_URL, PID_REPO_DIR])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", PID_REPO_DIR])

os.chdir(PID_REPO_DIR)
sys.path.insert(0, PID_REPO_DIR)

snapshot_download(
    repo_id="nvidia/PiD",
    local_dir=PID_REPO_DIR,
    allow_patterns=[
        "checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/*",
        "checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/*",
        "checkpoints/ae.safetensors",
    ],
)

from pid._src.inference.checkpoint_registry import get_pid_checkpoint
from pid._src.inference.pipeline_registry import (
    decode_with_pipeline_vae,
    extract_latent,
    load_pipeline,
)
from pid._src.utils.model_loader import load_model_from_checkpoint

DTYPE               = torch.bfloat16
BACKBONE            = "zimage"
SR_SCALE            = 4
PID_INFERENCE_STEPS = 4
MAX_SEED            = 2**31 - 1

print("[pid] loading Z-Image pipeline...", flush=True)

from transformers import masking_utils as _mu

def _broadcasting_vmap_for_bhqkv(mask_function, bh_indices: bool = True):
    def wrapped(b, h, q, k):
        if bh_indices:
            return mask_function(
                b[:, None, None, None],
                h[None, :, None, None],
                q[None, None, :, None],
                k[None, None, None, :],
            )
        return mask_function(b, h, q[:, None], k[None, :])
    return wrapped

_mu._vmap_for_bhqkv = _broadcasting_vmap_for_bhqkv

import transformers.models.gemma2.modeling_gemma2 as _gm

_orig_gemma2_forward = _gm.Gemma2Model.forward

def _patched_gemma2_forward(self, *args, **kwargs):
    _orig_tt = torch.tensor
    dev = self.embed_tokens.weight.device
    def _tt(data, *a, **kw):
        kw.setdefault("device", dev)
        return _orig_tt(data, *a, **kw)
    torch.tensor = _tt
    try:
        return _orig_gemma2_forward(self, *args, **kwargs)
    finally:
        torch.tensor = _orig_tt

_gm.Gemma2Model.forward = _patched_gemma2_forward


pipeline, pipe_cfg = load_pipeline(BACKBONE, dtype=DTYPE)
pipeline.to("cuda")

print("[pid] loading TAEF1 (fast preview decoder)...", flush=True)
from diffusers import AutoencoderTiny

taef1 = AutoencoderTiny.from_pretrained(
    "madebyollin/taef1", torch_dtype=DTYPE, low_cpu_mem_usage=False
).to("cuda")
taef1.eval()


def _load_pid(ckpt_type: str):
    meta = get_pid_checkpoint(BACKBONE, ckpt_type)
    print(f"[pid] loading PiD decoder ({ckpt_type})...", flush=True)
    model, _ = load_model_from_checkpoint(
        experiment_name=meta.experiment,
        checkpoint_path=meta.checkpoint_path,
        config_file="pid/_src/configs/pid/config.py",
        enable_fsdp=False,
        strict=False,
    )
    model.eval()
    return model


pid_models = {
    "2k":     _load_pid("2k"),
    "2kto4k": _load_pid("2kto4k"),
}

print("[pid] loading FLUX.2-Klein pipeline...", flush=True)
from diffusers import Flux2KleinPipeline

klein_pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=DTYPE,
).to("cuda")
print("[pid] FLUX.2-Klein loaded.", flush=True)
print("[pid] ready", flush=True)

def _pick_pid_model(resolution: int):
    return pid_models["2kto4k"] if resolution > 512 else pid_models["2k"]


def _taef1_preview(packed_latent: torch.Tensor, H: int, W: int) -> Image.Image:
    with torch.no_grad():
        unpacked = extract_latent(pipeline, SimpleNamespace(images=packed_latent), pipe_cfg, H, W)
        scale = pipeline.vae.config.scaling_factor
        shift = getattr(pipeline.vae.config, "shift_factor", None) or 0.0
        denorm = unpacked.to(dtype=DTYPE) / scale + shift
        img = taef1.decode(denorm).sample
        img = (img.float().clamp(-1, 1) + 1) / 2
        arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pid_pixel_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (
        (x[0].float().clamp(-1, 1) + 1) * 127.5
    ).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)


def _pid_stream(
    pid_model,
    latent: torch.Tensor,
    baseline_01: torch.Tensor,
    sigma: float,
    caption: str,
    num_steps: int = PID_INFERENCE_STEPS,
):
    from contextlib import nullcontext

    B = 1
    lq_h, lq_w = baseline_01.shape[-2], baseline_01.shape[-1]
    img_h, img_w = lq_h * SR_SCALE, lq_w * SR_SCALE

    caption_embs, _ = pid_model._encode_text_raw([caption])
    caption_embs = caption_embs.to(**pid_model.tensor_kwargs)

    lq_video_or_image    = (baseline_01 * 2.0 - 1.0).to(dtype=DTYPE, device="cuda")
    lq_latent            = latent.to(dtype=DTYPE, device="cuda")
    degrade_sigma_tensor = torch.tensor([sigma], device="cuda", dtype=torch.float32)

    gen   = torch.Generator(device="cuda").manual_seed(0)
    noise = torch.randn(B, 3, img_h, img_w, device="cuda", generator=gen)

    t_list = pid_model._get_t_list(device=torch.device("cuda"), num_steps=num_steps)
    autocast_ctx = (
        torch.autocast("cuda", dtype=pid_model.autocast_dtype)
        if pid_model.autocast_dtype
        else nullcontext()
    )
    net = pid_model.net
    net.eval()
    timescale           = pid_model.fm_trainer.timescale
    student_sample_type = pid_model.config.student_sample_type
    prediction_type     = pid_model.config.prediction_type

    x = noise
    with torch.no_grad(), autocast_ctx:
        steps_total = len(t_list) - 1
        for step_idx, (t_cur, t_next) in enumerate(zip(t_list[:-1], t_list[1:])):
            t_cur_batch  = t_cur.expand(B)
            t_cur_scaled = t_cur_batch * timescale
            v_pred = net(
                x,
                t_cur_scaled,
                caption_embs,
                lq_video_or_image=lq_video_or_image,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma_tensor,
            )
            if t_next.item() > 0:
                if student_sample_type == "ode":
                    v_for_step = pid_model._net_output_to_velocity(
                        x, v_pred, t_cur_batch, prediction_type
                    )
                    dt = t_next - t_cur
                    x  = x + dt * v_for_step
                else:
                    x0_pred   = pid_model._velocity_to_x0(x, v_pred, t_cur_batch)
                    eps_infer = torch.randn(
                        x0_pred.shape, device=x0_pred.device,
                        dtype=x0_pred.dtype, generator=gen,
                    )
                    s = [B] + [1] * (x.ndim - 1)
                    t_next_bcast = t_next.reshape(1).expand(s)
                    x = (1.0 - t_next_bcast) * x0_pred + t_next_bcast * eps_infer
            else:
                x = pid_model._velocity_to_x0(x, v_pred, t_cur_batch)
            yield step_idx + 1, steps_total, x.clone()


def _resize_to_divisible(image: Image.Image, max_side: int = 1024, div: int = 16) -> Image.Image:
    w, h  = image.size
    scale = min(max_side / w, max_side / h, 1.0)
    nw    = max(div, (int(w * scale) // div) * div)
    nh    = max(div, (int(h * scale) // div) * div)
    return image.resize((nw, nh), Image.LANCZOS)


def _encode_image_to_latent(image_01: torch.Tensor) -> torch.Tensor:
    vae        = pipeline.vae
    image_norm = image_01 * 2.0 - 1.0
    with torch.no_grad():
        latent = vae.encode(image_norm.to(dtype=DTYPE, device="cuda")).latent_dist.sample()
        scale  = vae.config.scaling_factor
        shift  = getattr(vae.config, "shift_factor", None) or 0.0
        latent = (latent - shift) * scale
    return latent

import random
import threading
import queue as _queue


def _generate_core(
    prompt: str,
    num_inference_steps: int = 28,
    guidance_scale: float   = 5.0,
    seed: int               = 0,
    resolution: int         = 512,
    randomize_seed: bool    = False,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    if randomize_seed:
        seed = random.randint(0, 2**31 - 1)
    seed                = int(seed)
    num_inference_steps = int(num_inference_steps)
    H = W               = int(resolution)

    yield (
        gr.update(visible=True,  value=None, label="Generating Z-Image…"),
        gr.update(visible=False, value=None),
        gr.update(value=seed),
    )

    preview_q: "_queue.Queue" = _queue.Queue()
    _DONE = object()

    def streaming_cb(pipe, step_index, timestep, callback_kwargs):
        try:
            preview = _taef1_preview(callback_kwargs["latents"], H, W)
            preview_q.put((step_index, preview))
        except Exception as e:
            print(f"[pid] taef1 preview failed at step {step_index}: {e}", flush=True)
        return callback_kwargs

    def run_pipeline():
        gen_torch  = torch.Generator(device="cuda").manual_seed(int(seed))
        gen_kwargs = dict(
            prompt=prompt, height=H, width=W,
            num_inference_steps=num_inference_steps,
            guidance_scale=float(guidance_scale),
            num_images_per_prompt=1,
            output_type="latent",
            generator=gen_torch,
            callback_on_step_end=streaming_cb,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        gen_kwargs.update(pipe_cfg.extra_generate_kwargs)
        try:
            with torch.no_grad():
                out = pipeline(**gen_kwargs)
            preview_q.put((_DONE, out))
        except Exception as e:
            preview_q.put((_DONE, e))

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    raw_output = None
    while True:
        step_index, payload = preview_q.get()
        if step_index is _DONE:
            if isinstance(payload, Exception):
                raise payload
            raw_output = payload
            break
        label = f"Generating Z-Image — step {step_index + 1}/{num_inference_steps}"
        yield (
            gr.update(visible=True, value=payload, label=label),
            gr.update(visible=False),
            gr.update(),
        )

    thread.join()
    final_latent = extract_latent(pipeline, raw_output, pipe_cfg, H, W)

    yield (
        gr.update(visible=True, label="Decoding final Z-Image…"),
        gr.update(visible=False),
        gr.update(),
    )
    with torch.no_grad():
        baseline_01 = decode_with_pipeline_vae(pipeline, final_latent, pipe_cfg)
    zimage_img = Image.fromarray(
        (baseline_01[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    )

    torch.cuda.empty_cache()

    final_sigma = float(pipeline.scheduler.sigmas[-1].item())
    pid_img     = None
    pid_model   = _pick_pid_model(H)

    for k, total, x in _pid_stream(pid_model, final_latent, baseline_01, final_sigma, prompt):
        pid_img = _pid_pixel_to_pil(x)
        yield (
            gr.update(visible=True, value=pid_img, label=f"Upscaling with PiD — step {k}/{total}"),
            gr.update(visible=False),
            gr.update(),
        )

    yield (
        gr.update(visible=False, value=None),
        gr.update(visible=True,  value=(zimage_img, pid_img)),
        gr.update(),
    )


@spaces.GPU(duration=60)
def generate_large(*args, **kwargs):
    yield from _generate_core(*args, **kwargs)


@spaces.GPU(duration=90, size="xlarge")
def generate_xlarge(*args, **kwargs):
    yield from _generate_core(*args, **kwargs)


def generate(prompt, num_inference_steps, guidance_scale, seed, resolution, randomize_seed):
    fn = generate_xlarge if int(resolution) >= 1024 else generate_large
    yield from fn(prompt, num_inference_steps, guidance_scale, seed, resolution, randomize_seed)


def update_dimensions_on_upload(image: Image.Image):
    if image is None:
        return "_Upload an image to see its processed dimensions._"
    resized = _resize_to_divisible(image)
    ow, oh  = image.size
    nw, nh  = resized.size
    return (
        f"**Input:** {ow} × {oh} px  →  "
        f"**Processed:** {nw} × {nh} px  →  "
        f"**PiD output:** {nw * SR_SCALE} × {nh * SR_SCALE} px"
    )


def _i2i_generate_core(
    input_image: Image.Image,
    prompt: str,
    seed: int            = 0,
    randomize_seed: bool = True,
    guidance_scale: float = 1.0,
    steps: int           = 4,
):
    if input_image is None:
        raise gr.Error("Please upload an input image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt / description.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)

    input_image = _resize_to_divisible(input_image.convert("RGB"))
    W, H = input_image.size

    yield (
        gr.update(visible=True,  value=None, label="Running FLUX.2-Klein…"),
        gr.update(visible=False, value=None),
        gr.update(value=seed),
    )

    gen_torch = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        klein_out = klein_pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=gen_torch,
            output_type="pil",
        )
    klein_img: Image.Image = klein_out.images[0]

    if klein_img.size != (W, H):
        klein_img = klein_img.resize((W, H), Image.LANCZOS)

    yield (
        gr.update(visible=True, value=klein_img, label="FLUX.2-Klein done — encoding for PiD…"),
        gr.update(visible=False),
        gr.update(),
    )

    torch.cuda.empty_cache()

    klein_arr       = np.array(klein_img).astype(np.float32) / 255.0
    klein_tensor_01 = torch.from_numpy(klein_arr).permute(2, 0, 1).unsqueeze(0)

    final_latent = _encode_image_to_latent(klein_tensor_01)
    baseline_01  = klein_tensor_01.to(dtype=DTYPE, device="cuda")
    final_sigma  = float(pipeline.scheduler.sigmas[-1].item())

    pid_model = _pick_pid_model(max(H, W))
    pid_img   = None

    for k, total, x in _pid_stream(
        pid_model, final_latent, baseline_01, final_sigma, prompt,
        num_steps=PID_INFERENCE_STEPS,
    ):
        pid_img = _pid_pixel_to_pil(x)
        yield (
            gr.update(visible=True, value=pid_img, label=f"Upscaling with PiD — step {k}/{total}"),
            gr.update(visible=False),
            gr.update(),
        )

    yield (
        gr.update(visible=False, value=None),
        gr.update(visible=True,  value=(klein_img, pid_img)),
        gr.update(),
    )


@spaces.GPU(duration=90, size="xlarge")
def i2i_generate(*args, **kwargs):
    yield from _i2i_generate_core(*args, **kwargs)


UPSCALER_MAX_SIDE = 1024


def _upscaler_dim_info(image: Image.Image):
    if image is None:
        return "_Upload an image to see its upscale dimensions._"
    w, h   = image.size
    scale  = min(UPSCALER_MAX_SIDE / w, UPSCALER_MAX_SIDE / h, 1.0)
    nw     = max(16, (int(w * scale) // 16) * 16)
    nh     = max(16, (int(h * scale) // 16) * 16)
    out_w, out_h = nw * SR_SCALE, nh * SR_SCALE
    return (
        f"**Input:** {w} × {h} px  →  "
        f"**Processed:** {nw} × {nh} px  →  "
        f"**Upscaled output:** {out_w} × {out_h} px  "
        f"*({SR_SCALE}× via PiD)*"
    )


def _upscaler_core(input_image: Image.Image, prompt: str):
    if input_image is None:
        raise gr.Error("Please upload an image to upscale.")

    caption = prompt.strip() if prompt and prompt.strip() else "high quality, detailed, sharp"

    img_rgb = input_image.convert("RGB")
    w, h    = img_rgb.size
    scale   = min(UPSCALER_MAX_SIDE / w, UPSCALER_MAX_SIDE / h, 1.0)
    nw      = max(16, (int(w * scale) // 16) * 16)
    nh      = max(16, (int(h * scale) // 16) * 16)
    if (nw, nh) != (w, h):
        img_rgb = img_rgb.resize((nw, nh), Image.LANCZOS)

    input_pil = img_rgb

    yield (
        gr.update(visible=True, value=input_pil, label="Encoding image…"),
        gr.update(visible=False, value=None),
    )

    arr_01    = np.array(img_rgb).astype(np.float32) / 255.0
    tensor_01 = torch.from_numpy(arr_01).permute(2, 0, 1).unsqueeze(0)

    latent      = _encode_image_to_latent(tensor_01)
    baseline_01 = tensor_01.to(dtype=DTYPE, device="cuda")
    sigma       = float(pipeline.scheduler.sigmas[-1].item())

    torch.cuda.empty_cache()

    pid_model = _pick_pid_model(max(nw, nh))
    pid_img   = None

    for k, total, x in _pid_stream(
        pid_model, latent, baseline_01, sigma, caption,
        num_steps=PID_INFERENCE_STEPS,
    ):
        pid_img = _pid_pixel_to_pil(x)
        yield (
            gr.update(visible=True, value=pid_img, label=f"Upscaling with PiD — step {k}/{total}"),
            gr.update(visible=False),
        )

    yield (
        gr.update(visible=False, value=None),
        gr.update(visible=True, value=(input_pil, pid_img)),
    )


@spaces.GPU(duration=90, size="xlarge")
def upscaler_run(*args, **kwargs):
    yield from _upscaler_core(*args, **kwargs)

DESCRIPTION = """
<div class="pid-header">
  <h1>PiD — <span class="accent">Pixel Diffusion</span> Decoder</h1>
  <div class="pid-tipbar">
    <strong>Text2Image</strong> —
    <a href="https://huggingface.co/Tongyi-MAI/Z-Image" target="_blank">Z-Image</a>
    with live TAEF1 previews →
    <a href="https://huggingface.co/nvidia/PiD" target="_blank">PiD</a>
    4-step pixel-diffusion 4× SR &nbsp;·&nbsp;
    <strong>Image2Image</strong> — FLUX.2-Klein → PiD 4× &nbsp;·&nbsp;
    <strong>Upscaler</strong> — PiD direct 4× &nbsp;·&nbsp;
    <a href="https://github.com/PRITHIVSAKTHIUR/PiD-Image-Upscaler" target="_blank">GitHub ↗</a>
  </div>
</div>
"""

with gr.Blocks(theme=DARK, css=css) as demo:

    gr.HTML(DESCRIPTION)

    with gr.Tabs():

        # ── Tab 1 : Image2Image ──────────────────────────────────────────
        with gr.Tab("Image2Image PiD"):

            gr.Markdown(
                "Upload any image — "
                "**[FLUX.2-Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** "
                "refines it then **PiD** super-resolves the result **4×**.  \n"
                "The slider compares the Klein output **(left)** to the PiD upscale **(right)**."
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    i2i_input    = gr.Image(label="Input image", type="pil", height=340)
                    i2i_dim_info = gr.Markdown(
                        "_Upload an image to see its processed dimensions._",
                        elem_classes=["pid-dim-badge"],
                    )
                    i2i_prompt = gr.Textbox(
                        label="Prompt / description",
                        placeholder="Describe the image content or the desired style…",
                        lines=3,
                    )
                    i2i_run = gr.Button("Run Image2Image", variant="primary", size="lg")

                    with gr.Accordion("Advanced Settings", open=False):
                        i2i_seed = gr.Slider(
                            label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0
                        )
                        i2i_rand     = gr.Checkbox(label="Randomize seed", value=True)
                        i2i_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=0.0, maximum=10.0, step=0.1, value=1.0,
                        )
                        i2i_steps = gr.Slider(
                            label="Steps", minimum=1, maximum=50, value=4, step=1
                        )

                with gr.Column(scale=2, min_width=340):
                    i2i_live = gr.Image(
                        label="Output preview",
                        visible=True, show_label=True,
                        type="pil", height=380,
                    )
                    i2i_slider = gr.ImageSlider(
                        label="FLUX.2-Klein (left)  ↔  PiD 4× upscale (right)",
                        visible=False,
                        type="pil",
                        height=680,
                        max_height=680,
                    )

            i2i_input.upload(
                fn=update_dimensions_on_upload,
                inputs=i2i_input,
                outputs=i2i_dim_info,
            )
            i2i_run.click(
                fn=i2i_generate,
                inputs=[i2i_input, i2i_prompt, i2i_seed, i2i_rand, i2i_guidance, i2i_steps],
                outputs=[i2i_live, i2i_slider, i2i_seed],
            )

        # ── Tab 2 : Text2Image ───────────────────────────────────────────
        with gr.Tab("Text2Image PiD"):

            with gr.Row():
                prompt = gr.Textbox(
                    show_label=False,
                    placeholder="Describe what you want to generate…",
                    value=(
                        "A photorealistic Labrador retriever resting beside a campfire at night, "
                        "glowing warm firelight reflecting on detailed fur, cinematic outdoor atmosphere."
                    ),
                    max_lines=1,
                    scale=4,
                    container=False,
                )
                run = gr.Button("Generate", variant="primary", scale=1)

            live_preview = gr.Image(
                label="Z-Image with PiD",
                visible=True, show_label=True,
                type="pil", height=680,
            )
            slider = gr.ImageSlider(
                label="Z-Image (left)  ↔  PiD 4× upscale (right)",
                visible=False,
                type="pil",
                height=680,
                max_height=680,
            )

            with gr.Accordion("Advanced settings", open=False):
                with gr.Row():
                    resolution = gr.Radio(
                        label="Z-Image resolution",
                        choices=[512, 1024],
                        value=512,
                        info="512 → 2048² (PiD 2k)  ·  1024 → 4096² (PiD 2kto4k)",
                    )
                    num_inference_steps = gr.Slider(
                        label="Z-Image steps",
                        minimum=8, maximum=50, step=1, value=28,
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance",
                        minimum=1.0, maximum=10.0, step=0.5, value=5.0,
                    )
                    seed           = gr.Number(label="Seed", value=0, precision=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            run.click(
                fn=generate,
                inputs=[prompt, num_inference_steps, guidance_scale, seed, resolution, randomize_seed],
                outputs=[live_preview, slider, seed],
            )

        # ── Tab 3 : Upscaler ─────────────────────────────────────────────
        with gr.Tab("Image Upscaler"):

            gr.Markdown(
                "Upload any image and **PiD** will upscale it **4×** directly — "
                "no text generation step needed.  \n"
                "An optional prompt / description helps PiD produce sharper, "
                "more faithful detail.  \n"
                "The slider compares the **original** *(left)* to the **PiD 4× upscale** *(right)*."
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    up_input = gr.Image(
                        label="Image to upscale",
                        type="pil", height=360,
                    )
                    up_dim_info = gr.Markdown(
                        "_Upload an image to see its upscale dimensions._",
                        elem_classes=["pid-dim-badge"],
                    )
                    up_prompt = gr.Textbox(
                        label="Optional prompt / description",
                        placeholder="Describe the image for better detail (leave blank for auto)…",
                        lines=3,
                        visible=False,
                    )
                    up_run = gr.Button("Upscale 4x", variant="primary", size="lg")

                with gr.Column(scale=2, min_width=340):
                    up_live = gr.Image(
                        label="Output preview",
                        visible=True, show_label=True,
                        type="pil", height=380,
                    )
                    up_slider = gr.ImageSlider(
                        label="Original (left)  ↔  PiD 4× upscale (right)",
                        visible=False,
                        type="pil",
                        height=680,
                        max_height=680,
                    )

            up_input.upload(
                fn=_upscaler_dim_info,
                inputs=up_input,
                outputs=up_dim_info,
            )
            up_run.click(
                fn=upscaler_run,
                inputs=[up_input, up_prompt],
                outputs=[up_live, up_slider],
            )

if __name__ == "__main__":
    demo.queue().launch(mcp_server=True, ssr_mode=False, show_error=True)
