"""Frozen image embeddings for aligned headshots.

SigLIP 2 base is the default; DINOv3 is the challenger. Embed both, train the
same heads on each, keep whichever wins held-out agreement.

    python -m sinemart.embed crops/ siglip2 out/siglip2.npz
    python -m sinemart.embed crops/ dinov3  out/dinov3.npz

Crops are named ``<applicant_id>-<kind>.jpg`` (or plain ``<applicant_id>.jpg``).
One person can have several usable face crops (closeup, full-body shots); we
embed each and average them into one vector per person, so a single odd
photo moves the score less. Output: ``ids`` (int64, one per applicant),
``emb`` (float32, L2-normalised mean), ``n_photos`` (crops averaged),
``model`` (str).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

MODELS = {
    "siglip2": "google/siglip2-base-patch16-224",
    # Gated on Hugging Face: accept the licence on the model page first.
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    # Open stand-in for DINOv3 until its licence is accepted.
    "dinov2": "facebook/dinov2-base",
}


def load(name: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    repo = MODELS[name]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).eval().to(device)

    @torch.inference_mode()
    def encode(images):
        x = proc(images=images, return_tensors="pt").to(device)
        out = model.get_image_features(**x) if name == "siglip2" else model(**x)
        # transformers 5 returns a model output; 4.x returned the tensor.
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output  # CLS for DINOv3
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy()

    return encode


CROP_RE = re.compile(r"^(\d+)(?:-[a-z0-9]+)?$")


def per_person(ids: list[int], emb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average each applicant's photo embeddings and re-normalise."""
    people = sorted(set(ids))
    idx = {a: i for i, a in enumerate(people)}
    total = np.zeros((len(people), emb.shape[1]), np.float64)
    n = np.zeros(len(people), np.int64)
    for a, e in zip(ids, emb):
        total[idx[a]] += e
        n[idx[a]] += 1
    total /= np.linalg.norm(total, axis=1, keepdims=True)
    return np.asarray(people, np.int64), total.astype(np.float32), n


def embed_dir(crop_dir: Path, name: str, batch: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from PIL import Image

    paths = sorted(p for p in crop_dir.glob("*.jpg") if CROP_RE.match(p.stem))
    encode = load(name)
    ids, chunks = [], []
    for i in range(0, len(paths), batch):
        group = paths[i : i + batch]
        chunks.append(encode([Image.open(p).convert("RGB") for p in group]))
        ids.extend(int(CROP_RE.match(p.stem)[1]) for p in group)
        print(f"\r{name}: {i + len(group)}/{len(paths)} photos", end="", flush=True)
    print()
    return per_person(ids, np.concatenate(chunks))


def main(argv: list[str]) -> None:
    if len(argv) != 4 or argv[2] not in MODELS:
        sys.exit(f"usage: python -m sinemart.embed CROP_DIR {{{','.join(MODELS)}}} OUT.npz")
    ids, emb, n = embed_dir(Path(argv[1]), argv[2])
    np.savez(argv[3], ids=ids, emb=emb, n_photos=n, model=MODELS[argv[2]])
    print(f"saved {emb.shape} to {argv[3]} ({n.sum()} face crops, up to {n.max()} per person)")


if __name__ == "__main__":
    main(sys.argv)
