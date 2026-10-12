"""Group applicants into cewek / cowok so battles and hills stay within one group.

The COI spreadsheet has no gender column, so we guess zero-shot with SigLIP 2:
each usable face crop is compared with the text "a photo of a woman" / "a photo
of a man" and a person's guess is the average over their crops. Wrong guesses
are fixed by hand on the scores page. Gender only groups people; it never
feeds into the score.

    python -m sinemart.gender CROP_DIR OUT.csv      # applicant_id,gender,p_female
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from .embed import CROP_RE, MODELS

PROMPTS = {
    "f": ["a photo of a woman", "a photo of a young woman", "a photo of a girl"],
    "m": ["a photo of a man", "a photo of a young man", "a photo of a boy"],
}


def p_female(img_emb: np.ndarray, text_f: np.ndarray, text_m: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """Softmax over the two classes from cosine similarity (inputs L2-normalised)."""
    sf = (img_emb @ text_f.T).max(axis=1) * scale
    sm = (img_emb @ text_m.T).max(axis=1) * scale
    return 1 / (1 + np.exp(sm - sf))


def guess(crop_dir: Path) -> dict[int, float]:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    repo = MODELS["siglip2"]
    proc = AutoProcessor.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).eval()
    norm = lambda t: torch.nn.functional.normalize(  # noqa: E731
        (t if isinstance(t, torch.Tensor) else t.pooler_output).float(), dim=-1).numpy()
    with torch.inference_mode():
        texts = {}
        for g, prompts in PROMPTS.items():
            x = proc(text=prompts, padding="max_length", max_length=64, return_tensors="pt")
            texts[g] = norm(model.get_text_features(**x))
        paths = sorted(p for p in crop_dir.glob("*.jpg") if CROP_RE.match(p.stem))
        per: dict[int, list[float]] = {}
        for i in range(0, len(paths), 32):
            group = paths[i : i + 32]
            x = proc(images=[Image.open(p).convert("RGB") for p in group], return_tensors="pt")
            pf = p_female(norm(model.get_image_features(**x)), texts["f"], texts["m"])
            for p, v in zip(group, pf):
                per.setdefault(int(CROP_RE.match(p.stem)[1]), []).append(float(v))
    return {a: float(np.mean(v)) for a, v in per.items()}


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        sys.exit("usage: python -m sinemart.gender CROP_DIR OUT.csv")
    probs = guess(Path(argv[1]))
    with open(argv[2], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["applicant_id", "gender", "p_female"])
        for a, p in sorted(probs.items()):
            w.writerow([a, "f" if p >= 0.5 else "m", f"{p:.3f}"])
    nf = sum(p >= 0.5 for p in probs.values())
    unsure = sum(0.2 < p < 0.8 for p in probs.values())
    print(f"{nf} cewek, {len(probs) - nf} cowok, {unsure} unsure (check these on the scores page)")


if __name__ == "__main__":
    main(sys.argv)
