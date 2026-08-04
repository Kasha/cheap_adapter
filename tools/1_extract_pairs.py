"""
Step 1: Extract paired embeddings from the SAME images with both models.

- MobileCLIP-S1 (phone-tier image encoder), via open_clip
- SigLIP 2 base (server-tier), via transformers
- Also extract SigLIP TEXT embeddings for the captions (for retrieval eval)

Output: pairs.npz
  mob_img:  [N, d_mob]   MobileCLIP image embeddings (L2-normalized)
  sig_img:  [N, d_sig]   SigLIP image embeddings     (L2-normalized)
  sig_txt:  [N, d_sig]   SigLIP text embeddings of caption #0
"""

import io
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_IMAGES = 4000          # 3000 train / 1000 eval split happens later
BATCH = 64

SIGLIP_ID = "google/siglip2-base-patch16-224"
MOBILECLIP_NAME = "MobileCLIP-S1"
MOBILECLIP_PRETRAINED = "datacompdr"


def load_dataset_pairs(n):
    """Yields (PIL image, caption) from COCO val2017.
    Downloads directly from the official COCO server (plain zips, no HF
    loading scripts — immune to the datasets-library scripts deprecation).
    ~1GB images + 240MB annotations, downloaded once and cached."""
    import json
    import urllib.request
    import zipfile

    root = Path(os.environ.get("COCO_DIR", str(DATA_DIR / "coco")))
    root.mkdir(parents=True, exist_ok=True)
    img_dir = root / "val2017"
    ann_file = root / "annotations" / "captions_val2017.json"

    def fetch(url, dest_zip):
        if not dest_zip.exists():
            print(f"downloading {url} ...")
            urllib.request.urlretrieve(url, dest_zip)
        with zipfile.ZipFile(dest_zip) as z:
            z.extractall(root)

    if not ann_file.exists():
        fetch("http://images.cocodataset.org/annotations/"
              "annotations_trainval2017.zip", root / "ann.zip")
    if not img_dir.exists():
        fetch("http://images.cocodataset.org/zips/val2017.zip",
              root / "val2017.zip")

    ann = json.load(open(ann_file))
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    id2cap = {}
    for a in ann["annotations"]:  # keep first caption per image
        id2cap.setdefault(a["image_id"], a["caption"])

    count = 0
    for img_id, cap in id2cap.items():
        path = img_dir / id2file[img_id]
        if not path.exists():
            continue
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        yield img, cap
        count += 1
        if count >= n:
            break
    if count == 0:
        raise RuntimeError(
            "No images loaded — check that the COCO download completed "
            f"under {root}"
        )


@torch.no_grad()
def main():
    import open_clip
    from transformers import AutoModel, AutoProcessor

    print("Loading MobileCLIP-S1...")
    mob, _, mob_pre = open_clip.create_model_and_transforms(
        MOBILECLIP_NAME, pretrained=MOBILECLIP_PRETRAINED
    )
    mob.eval().to(DEVICE)

    print("Loading SigLIP 2...")
    sig = AutoModel.from_pretrained(SIGLIP_ID).eval().to(DEVICE)
    sig_proc = AutoProcessor.from_pretrained(SIGLIP_ID)

    mob_img, sig_img, sig_txt = [], [], []
    imgs_buf, caps_buf = [], []

    def as_tensor(x):
        """transformers version shim: get_*_features may return a bare
        tensor (older) or a ModelOutput object (newer)."""
        if torch.is_tensor(x):
            return x
        for attr in ("pooler_output", "image_embeds", "text_embeds",
                     "last_hidden_state"):
            v = getattr(x, attr, None)
            if torch.is_tensor(v):
                return v if v.dim() == 2 else v.mean(1)
        raise TypeError(f"Cannot extract tensor from {type(x)}")

    def flush():
        if not imgs_buf:
            return
        # MobileCLIP image
        mb = torch.stack([mob_pre(im) for im in imgs_buf]).to(DEVICE)
        e = as_tensor(mob.encode_image(mb))
        mob_img.append(torch.nn.functional.normalize(e, dim=-1)
                       .float().cpu().numpy())
        # SigLIP image + text
        si = sig_proc(images=imgs_buf, return_tensors="pt").to(DEVICE)
        e = as_tensor(sig.get_image_features(**si))
        sig_img.append(torch.nn.functional.normalize(e, dim=-1)
                       .float().cpu().numpy())
        st = sig_proc(text=caps_buf, return_tensors="pt",
                      padding="max_length", truncation=True,
                      max_length=64).to(DEVICE)
        e = as_tensor(sig.get_text_features(**st))
        sig_txt.append(torch.nn.functional.normalize(e, dim=-1)
                       .float().cpu().numpy())
        imgs_buf.clear()
        caps_buf.clear()

    print("Extracting pairs...")
    for i, (img, cap) in enumerate(load_dataset_pairs(N_IMAGES)):
        imgs_buf.append(img)
        caps_buf.append(cap)
        if len(imgs_buf) == BATCH:
            flush()
            if (i + 1) % (BATCH * 5) == 0:
                print(f"  {i + 1}/{N_IMAGES}")
    flush()

    mob_img = np.concatenate(mob_img)
    sig_img = np.concatenate(sig_img)
    sig_txt = np.concatenate(sig_txt)
    np.savez_compressed(str(DATA_DIR / "pairs.npz"), mob_img=mob_img,
                        sig_img=sig_img, sig_txt=sig_txt)
    print(f"Saved: mob_img {mob_img.shape}, sig_img {sig_img.shape}, "
          f"sig_txt {sig_txt.shape}")


if __name__ == "__main__":
    main()
