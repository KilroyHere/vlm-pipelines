"""Automatic masks for the SAM 3D spike: text prompt -> OWLv2 box -> SAM 2 mask.

In the product the box comes from RoomPlan (an object's 3D box projected into a captured
frame). The capture app doesn't exist yet, so an open-vocabulary detector stands in for the
projected box. Used by ``make_mask.py auto`` and ``fetch_sources.py``.

The pure parts (prompts, box selection, NMS) need only the standard library and are unit
tested. torch/transformers are imported inside the model classes only, so importing this
module stays light.

Box selection rule (``select_box``), documented in README section 3:
1. Score every detector box against the item's *target* queries (max over synonyms) and its
   *distractor* queries (e.g. "television" for a tv_stand). Drop boxes whose best distractor
   score beats their target score, boxes below ``min_score``, and boxes covering less than
   ``min_frac`` or more than ``max_frac`` of the image (whole-image boxes are a common OWL
   false positive).
2. Non-maximum suppression (IoU 0.5) on the remaining boxes.
3. "Confident" boxes: score >= ``rel_score`` x the best score. Among them pick the
   **largest** box (the evaluation photo is about one main object; a smaller confident box is
   usually a part, a cushion, or a second, farther instance). Ties: higher score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

DETECTOR_MODEL = "google/owlv2-base-patch16-ensemble"
SAM_MODEL = "facebook/sam2.1-hiera-large"

# category -> (target queries, distractor queries). The first target query is the prompt.
PROMPTS: dict[str, tuple[list[str], list[str]]] = {
    "sofa": (["sofa", "couch"], ["armchair", "coffee table"]),
    "sectional": (["sectional sofa", "couch", "sofa"], ["armchair", "coffee table"]),
    "armchair": (["armchair", "lounge chair"], ["sofa", "side table"]),
    "dining_chair": (["chair", "dining chair"], ["table"]),
    "office_chair": (["office chair", "desk chair"], ["desk"]),
    "coffee_table": (["coffee table"], ["sofa", "rug"]),
    "round_table": (["round table", "dining table"], ["chair"]),
    "dining_table": (["dining table", "table"], ["chair"]),
    "desk": (["desk", "writing desk"], ["chair", "computer monitor"]),
    "bed": (["bed"], ["pillow", "nightstand"]),
    "bookshelf": (["bookshelf", "bookcase"], ["book"]),
    "dresser": (["dresser", "chest of drawers"], []),
    "wardrobe": (["wardrobe", "closet"], ["door"]),
    "tv_stand": (["tv stand", "media console"], ["television"]),
}

MIN_SCORE = 0.10
REL_SCORE = 0.6
MIN_FRAC = 0.01
MAX_FRAC = 0.97
NMS_IOU = 0.5


def queries_for(category: str, prompt: str | None = None, subtype: str | None = None):
    """(target, distractor) text queries for an item. ``prompt`` (comma-separated) replaces
    the targets; ``subtype`` (e.g. meta "subtype": "office chair") is tried first."""
    targets, distractors = PROMPTS.get(category, ([category.replace("_", " ")], []))
    if prompt:
        targets = [p.strip() for p in prompt.split(",") if p.strip()]
    elif subtype:
        targets = [subtype] + [t for t in targets if t != subtype]
    distractors = [d for d in distractors if d not in targets]
    return list(targets), list(distractors)


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in image pixels
    score: float  # best target-query score
    label: str  # best target query
    distractor_score: float = 0.0

    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def to_json(self) -> dict:
        d = asdict(self)
        d["box"] = [round(v, 1) for v in self.box]
        d["score"] = round(self.score, 4)
        d["distractor_score"] = round(self.distractor_score, 4)
        return d


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets: list[Detection], thr: float = NMS_IOU) -> list[Detection]:
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(iou(d.box, k.box) < thr for k in keep):
            keep.append(d)
    return keep


def filter_boxes(
    dets: list[Detection],
    image_size: tuple[int, int],
    min_score: float = MIN_SCORE,
    min_frac: float = MIN_FRAC,
    max_frac: float = MAX_FRAC,
) -> list[Detection]:
    w, h = image_size
    out = []
    for d in dets:
        frac = d.area() / float(w * h)
        if d.score < min_score or d.distractor_score > d.score:
            continue
        if not (min_frac <= frac <= max_frac):
            continue
        out.append(d)
    return nms(out)


def select_box(
    dets: list[Detection],
    image_size: tuple[int, int],
    min_score: float = MIN_SCORE,
    rel_score: float = REL_SCORE,
    min_frac: float = MIN_FRAC,
    max_frac: float = MAX_FRAC,
) -> Detection | None:
    """The box rule in the module docstring. None when nothing qualifies."""
    kept = filter_boxes(dets, image_size, min_score, min_frac, max_frac)
    if not kept:
        return None
    best = max(d.score for d in kept)
    confident = [d for d in kept if d.score >= rel_score * best]
    return max(confident, key=lambda d: (d.area(), d.score))


def box_fraction(det: Detection, image_size: tuple[int, int]) -> float:
    return det.area() / float(image_size[0] * image_size[1])


# --------------------------------------------------------------------------- models


def pick_device(device: str | None = None) -> str:
    import torch

    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class OwlDetector:
    """OWLv2 open-vocabulary detector (HF transformers; Apache-2.0, not gated)."""

    def __init__(self, model: str = DETECTOR_MODEL, device: str | None = None):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = pick_device(device)
        self.name = model
        self.processor = Owlv2Processor.from_pretrained(model)
        self.model = Owlv2ForObjectDetection.from_pretrained(model).to(self.device).eval()

    def detect(self, image, targets: list[str], distractors: list[str]) -> list[Detection]:
        """All boxes with per-query scores (``image``: PIL RGB)."""
        torch = self.torch
        queries = [f"a photo of a {q}" for q in targets + distractors]
        inputs = self.processor(text=[queries], images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        probs = torch.sigmoid(out.logits[0]).float().cpu()  # (boxes, queries)
        cx, cy, bw, bh = out.pred_boxes[0].float().cpu().unbind(-1)
        # OWLv2 pads to a square (bottom/right), so boxes are relative to the padded side.
        w, h = image.size
        side = float(max(w, h))
        x0 = ((cx - bw / 2) * side).clamp(0, w)
        y0 = ((cy - bh / 2) * side).clamp(0, h)
        x1 = ((cx + bw / 2) * side).clamp(0, w)
        y1 = ((cy + bh / 2) * side).clamp(0, h)
        nt = len(targets)
        t_scores, t_idx = probs[:, :nt].max(dim=1)
        d_scores = probs[:, nt:].max(dim=1).values if distractors else torch.zeros(len(probs))
        keep = (t_scores >= 0.02).nonzero().flatten().tolist()
        return [
            Detection(
                (float(x0[i]), float(y0[i]), float(x1[i]), float(y1[i])),
                float(t_scores[i]),
                targets[int(t_idx[i])],
                float(d_scores[i]),
            )
            for i in keep
        ]


class Sam2Segmenter:
    """Box-prompted SAM 2.1 via the transformers port (Apache-2.0, not gated)."""

    def __init__(self, model: str = SAM_MODEL, device: str | None = None):
        import torch
        from transformers import Sam2Model, Sam2Processor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # "sam2_video checkpoint into Sam2Model" is expected
        self.torch = torch
        self.device = pick_device(device)
        self.name = model
        self.processor = Sam2Processor.from_pretrained(model)
        self.model = Sam2Model.from_pretrained(model).to(self.device).eval()

    def segment(self, image, box) -> tuple[object, float]:
        """(bool HxW numpy mask, predicted IoU score) for one box prompt."""
        torch = self.torch
        inputs = self.processor(
            images=image, input_boxes=[[[float(v) for v in box]]], return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu()
        )[0]
        mask = masks[0, 0].numpy() > 0
        return mask, float(out.iou_scores[0, 0, 0].float().cpu())
