#!/usr/bin/env python
"""Download the evaluation photos listed in photo_sources.md and pick one per item, without
anyone choosing images by hand.

    python fetch_sources.py --data data                 # all sourceable items
    python fetch_sources.py --data data --items 1,4,13  # by case number
    python fetch_sources.py --data data --reselect      # re-run selection, no network
    python fetch_sources.py --data data --no-detect     # download only

Per item (``<category>_<NN>``):
1. GET the listing page (one request at a time, a browser User-Agent that also names this
   tool, timeouts, robots.txt respected). Bot-protection pages (Cloudflare, AWS WAF, Vercel,
   HTTP 403/429) are skipped and the reason recorded; nothing is retried or worked around.
2. Candidate images: og:image, JSON-LD images, gallery ``<img>`` src/srcset (largest
   variant; inside ``<article>`` when the page has one). Logos, icons, avatars and
   measurement drawings are dropped. Downloads go to ``data/<item>/candidates/`` with
   ``index.json``; images under ``MIN_SIDE`` px are discarded.
3. Auto-select: OWLv2 on every candidate (``automask``); the image whose selected box (same
   rule as ``make_mask.py auto``) has the highest score wins, provided that box covers a
   sensible fraction of the image (``FRAC_RANGE``, stricter for product photos). Ties go to
   the earlier gallery position.
4. ``make_mask.prepare_item`` writes ``image.jpg`` (EXIF upright, <= 2048 px) and meta.json
   with ``source_url``, ``image_url`` and, when the page states W/D/H, ``dims`` +
   ``dims_provenance`` (``manufacturer`` for retailer pages, else ``seller_reported``).
   Without stated dims, dims stay unset (normalization keeps the mesh scale and warns).

Writes ``data/fetch_report.json``. Images are for internal evaluation only; ``data/`` is
gitignored. Only the URLs in photo_sources.md and the image URLs on those pages are fetched.
Detection needs torch + transformers; everything else is standard library.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

HERE = Path(__file__).resolve().parent
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/129.0 Safari/537.36 roomlayout-sam3d-spike/0.1 (internal evaluation)"
)
TIMEOUT_S = 30
DELAY_S = 1.0
MAX_IMAGE_BYTES = 20_000_000
MIN_SIDE = 400
MAX_CANDIDATES = 200
# Selected box must cover this fraction of the candidate image.
FRAC_RANGE = {"product": (0.15, 0.97), "in_room": (0.04, 0.95)}
MANUFACTURER_HOSTS = ("ikea.com",)

# EVAL.md section 1: case -> (category, condition tags)
CASES: dict[int, tuple[str, list[str]]] = {
    1: ("sofa", ["product"]),
    2: ("sofa", ["in_room"]),
    3: ("sectional", ["in_room"]),
    4: ("sectional", ["product"]),
    5: ("armchair", ["in_room", "occluded"]),
    6: ("armchair", ["in_room", "low_light"]),
    7: ("dining_chair", ["product"]),
    8: ("dining_chair", ["in_room", "occluded"]),
    9: ("office_chair", ["in_room"]),
    10: ("coffee_table", ["in_room", "clutter"]),
    11: ("round_table", ["in_room"]),
    12: ("dining_table", ["in_room", "occluded"]),
    13: ("desk", ["in_room", "clutter"]),
    14: ("bed", ["in_room"]),
    15: ("bookshelf", ["in_room"]),
    16: ("dresser", ["product"]),
    17: ("wardrobe", ["in_room"]),
    18: ("tv_stand", ["in_room", "occluded"]),
    19: ("sofa", ["in_room", "back_view"]),
    20: ("armchair", ["in_room", "odd_angle"]),
}

BLOCK_MARKERS = {
    "Cloudflare challenge": ("Just a moment...", "challenges.cloudflare.com", "cf-chl"),
    "AWS WAF challenge": ("awsWafCookieDomainList", "AwsWafIntegration"),
    "Vercel Security Checkpoint": ("Vercel Security Checkpoint",),
    "captcha": ("captcha-delivery.com", "g-recaptcha", "hcaptcha.com"),
}
SKIP_URL = re.compile(
    r"(logo|icon|favicon|avatar|sprite|placeholder|gravatar|emoji|badge|/orglogos/|"
    r"\.svg|\.gif|\.ico|data:)",
    re.I,
)
SKIP_ALT = re.compile(r"(measurement|illustration|diagram|dimension|logo|icon)", re.I)
IMG_EXT = re.compile(r"\.(jpe?g|png|webp)(\?|$)", re.I)
WP_SIZE = re.compile(r"-(\d{2,4})x(\d{2,4})(\.(?:jpe?g|png|webp))$", re.I)


@dataclass
class Source:
    case: int
    item_id: str
    category: str
    condition: list[str]
    url: str | None
    case_text: str = ""


@dataclass
class Candidate:
    url: str
    alt: str = ""
    origin: str = ""  # og | jsonld | img
    file: str = ""
    width: int = 0
    height: int = 0
    status: str = "pending"  # ok | too_small | download_failed | not_image
    detection: dict | None = None
    select_score: float | None = None
    reject: str = ""


@dataclass
class ItemReport:
    case: int
    item_id: str
    source_url: str | None
    fetched: bool = False
    reason: str = ""
    n_candidates: int = 0
    chosen_url: str = ""
    detector_score: float | None = None
    box_area_frac: float | None = None
    dims: dict | None = None
    dims_provenance: str = ""
    dims_text: str = ""
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- sources


def parse_sources(md: str) -> list[Source]:
    """Rows of the photo_sources.md table: | # | EVAL case | source URL | ..."""
    out = []
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        case = int(cells[0])
        category, condition = CASES[case]
        m = re.search(r"https?://\S+", cells[2])
        url = m.group(0) if m else None
        out.append(Source(case, f"{category}_{case:02d}", category, condition, url, cells[1]))
    return sorted(out, key=lambda s: s.case)


# --------------------------------------------------------------------------- HTML


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.og: list[str] = []
        self.jsonld: list[str] = []
        self.imgs: list[tuple[str, str, bool]] = []  # (url, alt, inside article)
        self.title = ""
        self._in_ld = False
        self._in_title = False
        self._buf: list[str] = []
        self._article = 0

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag == "meta" and a.get("property", a.get("name", "")) in (
            "og:image",
            "og:image:url",
            "og:image:secure_url",
        ):
            self.og.append(a.get("content", ""))
        elif tag == "script" and "ld+json" in a.get("type", ""):
            self._in_ld, self._buf = True, []
        elif tag == "title":
            self._in_title = True
        elif tag == "article" or (tag in ("div", "main") and "entry-content" in a.get("class", "")):
            self._article += 1
        elif tag in ("img", "source"):
            alt = a.get("alt", "")
            inside = self._article > 0
            for key in ("srcset", "data-srcset", "data-lazy-srcset"):
                if a.get(key):
                    best = largest_from_srcset(a[key])
                    if best:
                        self.imgs.append((best, alt, inside))
            for key in ("data-src", "data-lazy-src", "data-orig-file", "src"):
                if a.get(key) and not a[key].startswith("data:"):
                    self.imgs.append((a[key], alt, inside))

    def handle_endtag(self, tag):
        if tag == "script" and self._in_ld:
            self._in_ld = False
            self.jsonld.append("".join(self._buf))
        elif tag == "title":
            self._in_title = False
        elif tag in ("article",) and self._article:
            self._article -= 1

    def handle_data(self, data):
        if self._in_ld:
            self._buf.append(data)
        elif self._in_title:
            self.title += data


def largest_from_srcset(srcset: str) -> str | None:
    best, best_w = None, -1.0
    for part in srcset.split(","):
        bits = part.strip().split()
        if not bits:
            continue
        w = 0.0
        if len(bits) > 1 and bits[1][-1:] in ("w", "x"):
            try:
                w = float(bits[1][:-1])
            except ValueError:
                w = 0.0
        if w > best_w:
            best, best_w = bits[0], w
    return best


def _jsonld_images(node, out: list[str]) -> None:
    if isinstance(node, list):
        for x in node:
            _jsonld_images(x, out)
    elif isinstance(node, dict):
        types = node.get("@type")
        types = types if isinstance(types, list) else [types]
        if "ImageObject" in types:
            for k in ("contentUrl", "url"):
                if isinstance(node.get(k), str):
                    out.append(node[k])
                    break
        img = node.get("image")
        if isinstance(img, str):
            out.append(img)
        elif img is not None:
            _jsonld_images(img, out)
        for k, v in node.items():
            if k != "image" and isinstance(v, list | dict) and k in ("@graph", "itemListElement"):
                _jsonld_images(v, out)


def jsonld_objects(blocks: list[str]) -> list:
    objs = []
    for b in blocks:
        try:
            objs.append(json.loads(b))
        except json.JSONDecodeError:
            continue
    return objs


def canonical_key(url: str) -> str:
    """Dedup key: drop the query and a WordPress -WxH size suffix."""
    path = urlsplit(url)._replace(query="", fragment="").geturl()
    return WP_SIZE.sub(r"\3", path)


def extract_candidates(page_html: str, base_url: str) -> list[Candidate]:
    p = _PageParser()
    p.feed(page_html)
    raw: list[tuple[str, str, str]] = [(u, "", "og") for u in p.og]
    ld: list[str] = []
    _jsonld_images(jsonld_objects(p.jsonld), ld)
    raw += [(u, "", "jsonld") for u in ld]
    imgs = p.imgs
    if any(inside for _, _, inside in imgs):  # blog/article page: only the article's images
        imgs = [i for i in imgs if i[2]]
    raw += [(u, alt, "img") for u, alt, _ in imgs]
    return filter_candidates(raw, base_url)


def filter_candidates(raw: list[tuple[str, str, str]], base_url: str) -> list[Candidate]:
    """Resolve, drop non-photos, and dedup (keeping the first position and any alt text)."""
    by_key: dict[str, Candidate] = {}
    for url, alt, origin in raw:
        url = urljoin(base_url, html.unescape(url.strip()))
        if not url.startswith(("http://", "https://")) or SKIP_URL.search(url):
            continue
        if origin == "img" and not IMG_EXT.search(url):
            continue
        m = WP_SIZE.search(urlsplit(url).path)
        if m and max(int(m.group(1)), int(m.group(2))) < MIN_SIDE:
            continue  # thumbnail variant; the full-size one is usually also listed
        key = canonical_key(url)
        if key in by_key:
            c = by_key[key]
            c.alt = c.alt or alt
            continue
        by_key[key] = Candidate(url=url, alt=alt, origin=origin)
    return [c for c in by_key.values() if not SKIP_ALT.search(c.alt)]


# --------------------------------------------------------------------------- dimensions

_NUM = r"(\d+(?:\.\d+)?(?:\s+\d+/\d+)?|\d+/\d+)"
_FRAC_CHARS = {"½": " 1/2", "¼": " 1/4", "¾": " 3/4", "⅛": " 1/8", "⅜": " 3/8", "⅝": " 5/8",
               "⅞": " 7/8", "⅓": " 1/3", "⅔": " 2/3"}
_UNIT = r"(\"|”|''|in\b|inch(?:es)?\b|cm\b|mm\b|m\b|'|ft\b|feet\b)"
TO_M = {"in": 0.0254, "cm": 0.01, "mm": 0.001, "m": 1.0, "ft": 0.3048}


def _norm_text(s: str) -> str:
    for k, v in _FRAC_CHARS.items():
        s = s.replace(k, v)
    return s.replace("&quot;", '"').replace("″", '"').replace("′", "'").replace("×", "x")


def _num(s: str) -> float:
    total = Fraction(0)
    for part in s.split():
        total += Fraction(part)
    return float(total)


def _unit(u: str | None, default: str = "in") -> str:
    if not u:
        return default
    u = u.lower()
    if u in ('"', "”", "''") or u.startswith("in"):
        return "in"
    if u in ("'", "ft", "feet"):
        return "ft"
    return u


def to_meters(value: str, unit: str | None, default: str = "in") -> float:
    return round(_num(value) * TO_M[_unit(unit, default)], 4)


def parse_dims_text(text: str) -> tuple[dict, str] | None:
    """W/D/H from free listing text. Returns ({width, depth, height} in m, matched text).

    Handles e.g. ``W 84" x D 38" x H 33"``, ``84"W x 38"D x 33"H``, ``84 x 38 x 33 in``,
    ``213 x 97 x 84 cm`` (assumed W x D x H), ``Width: 84 in ... Depth: 38 in ...`` and
    feet (``7' x 3'2" x 2'9"`` is not attempted; plain ``7 ft``/``6.5'`` is)."""
    t = _norm_text(text)
    lab = {"w": "width", "d": "depth", "l": "width", "h": "height"}
    # 1) letter-labelled triples, letter before or after the number
    pre = rf"\b([WDHL])\s*[:.]?\s*{_NUM}\s*{_UNIT}?"
    post = rf"{_NUM}\s*{_UNIT}?\s*([WDHL])\b"
    for pat, li, ni, ui in ((pre, 0, 1, 2), (post, 2, 0, 1)):
        sep = r"\s*(?:x|by|,|/)\s*"
        for m in re.finditer(pat + sep + pat + sep + pat, t, re.I):
            g = m.groups()
            per = len(g) // 3
            vals, units = {}, [g[i * per + ui] for i in range(3)]
            unit = next((u for u in reversed(units) if u), None)
            for i in range(3):
                key = lab[g[i * per + li].lower()]
                vals[key] = to_meters(g[i * per + ni], g[i * per + ui] or unit)
            if len(vals) == 3:
                return vals, m.group(0)
    # 2) "Width: 84 in" style labels
    found = {}
    for key, words in (
        ("width", r"width|wide|length"),
        ("depth", r"depth|deep"),
        ("height", r"height|tall|high"),
    ):
        m = re.search(rf"\b(?:overall\s+)?(?:{words})\b\s*[:=]?\s*{_NUM}\s*{_UNIT}", t, re.I)
        if not m:
            m = re.search(rf"{_NUM}\s*{_UNIT}\s*(?:{words})\b", t, re.I)
        if m:
            found[key] = (to_meters(m.group(1), m.group(2)), m.group(0))
    if len(found) == 3:
        return {k: v[0] for k, v in found.items()}, "; ".join(v[1] for v in found.values())
    # 3) bare triple with a unit: 84 x 38 x 33 in  (W x D x H)
    m = re.search(rf"{_NUM}\s*{_UNIT}?\s*x\s*{_NUM}\s*{_UNIT}?\s*x\s*{_NUM}\s*{_UNIT}", t, re.I)
    if m:
        g = m.groups()
        unit = g[5]
        vals = [to_meters(g[i], g[i + 1] or unit) for i in (0, 2, 4)]
        return dict(zip(("width", "depth", "height"), vals, strict=True)), m.group(0)
    return None


def parse_ikea_dims(page_html: str) -> tuple[dict, str] | None:
    """IKEA product pages: JSON-LD Product width/depth/height, else the Measurements
    accordion (Max. width / Max. depth for sectionals), never the packaging section."""
    for obj in jsonld_objects(_PageParser.jsonld_blocks(page_html)):
        for o in obj if isinstance(obj, list) else [obj]:
            if isinstance(o, dict) and o.get("@type") == "Product":
                vals = [o.get(k) for k in ("width", "depth", "height")]
                if all(isinstance(v, str) for v in vals):
                    try:
                        m = [_ikea_value(v) for v in vals]
                    except ValueError:
                        continue
                    text = f"JSON-LD width {vals[0]}, depth {vals[1]}, height {vals[2]}"
                    return dict(zip(("width", "depth", "height"), m, strict=True)), text
    i = page_html.find('id="SEC_measurements')
    if i < 0:
        i = page_html.find(">Measurements<")
    if i < 0:
        return None
    seg = page_html[i : i + 20000]
    j = seg.find("measurements-packaging", 50)
    seg = seg[:j] if j > 0 else seg
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " | ", seg)))
    pairs = re.findall(r"\|\s*([A-Za-z. ()]+?)\s*\|\s*([\d ½¼¾⅛⅜⅝⅞/.]+\s*(?:\"|cm|in)?)", text)
    table = {k.strip().lower(): v for k, v in pairs}
    pick = {
        "width": ("width", "max. width"),
        "depth": ("max. depth", "depth"),
        "height": ("height", "max. height"),
    }
    out = {}
    for key, labels in pick.items():
        for lab in labels:
            if lab in table:
                out[key] = (_ikea_value(table[lab]), f"{lab} {table[lab].strip()}")
                break
    if len(out) == 3:
        return {k: v[0] for k, v in out.items()}, "Measurements: " + ", ".join(
            v[1] for v in out.values()
        )
    return None


def _ikea_value(v: str) -> float:
    v = _norm_text(v).strip()
    m = re.fullmatch(rf"{_NUM}\s*{_UNIT}?", v)
    if not m:
        raise ValueError(v)
    return to_meters(m.group(1), m.group(2), default="in")


_PageParser.jsonld_blocks = staticmethod(  # type: ignore[attr-defined]
    lambda h: re.findall(r"<script[^>]+ld\+json[^>]*>(.*?)</script>", h, re.S)
)


def page_text(page_html: str) -> str:
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page_html, flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body)))


def dims_from_page(page_html: str, url: str) -> tuple[dict, str, str] | None:
    """(dims in m, provenance, matched text), or None when the page states no W/D/H."""
    host = urlsplit(url).hostname or ""
    manufacturer = any(host.endswith(h) for h in MANUFACTURER_HOSTS)
    if "ikea.com" in host:
        r = parse_ikea_dims(page_html)
        if r:
            return r[0], "manufacturer", r[1]
    r = parse_dims_text(page_text(page_html))
    if r and all(0.05 <= v <= 6 for v in r[0].values()):
        return r[0], "manufacturer" if manufacturer else "seller_reported", r[1]
    return None


# --------------------------------------------------------------------------- network


def block_reason(status: int, body: str) -> str | None:
    for name, marks in BLOCK_MARKERS.items():
        if any(m in body for m in marks):
            return f"HTTP {status}, bot protection ({name})"
    if status in (401, 403, 429):
        return f"HTTP {status} (access refused)"
    if status == 202 and len(body) < 5000:
        return "HTTP 202 with an empty interstitial page (bot challenge)"
    if status >= 400:
        return f"HTTP {status}"
    return None


class Fetcher:
    """Sequential, polite HTTP: one request at a time, delay, timeout, robots.txt."""

    def __init__(self, delay: float = DELAY_S):
        self.delay = delay
        self._last = 0.0
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _wait(self):
        dt = time.monotonic() - self._last
        if dt < self.delay:
            time.sleep(self.delay - dt)
        self._last = time.monotonic()

    def _raw(self, url: str, accept: str, max_bytes: int) -> tuple[int, bytes, str]:
        self._wait()
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": accept,
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return r.status, r.read(max_bytes + 1), r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            return e.code, e.read(200_000), e.headers.get("Content-Type", "")

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                status, body, _ = self._raw(root + "/robots.txt", "text/plain", 1_000_000)
                rp.parse(body.decode("utf-8", "replace").splitlines() if status == 200 else [])
                self._robots[root] = rp
            except Exception:
                self._robots[root] = None  # unreachable robots.txt: treat as allowed
        rp = self._robots[root]
        return rp is None or rp.can_fetch(USER_AGENT, url)

    def page(self, url: str) -> tuple[int, str]:
        status, body, _ = self._raw(url, "text/html,application/xhtml+xml", 10_000_000)
        return status, body.decode("utf-8", "replace")

    def image(self, url: str) -> bytes:
        status, body, ctype = self._raw(url, "image/avif,image/webp,image/*,*/*", MAX_IMAGE_BYTES)
        if status != 200:
            raise OSError(f"HTTP {status}")
        if len(body) > MAX_IMAGE_BYTES:
            raise OSError("larger than MAX_IMAGE_BYTES")
        if ctype and not ctype.startswith("image/"):
            raise OSError(f"content-type {ctype}")
        return body


# --------------------------------------------------------------------------- selection


def frac_range(condition: list[str]) -> tuple[float, float]:
    return FRAC_RANGE["product" if "product" in condition else "in_room"]


def choose_candidate(cands: list[Candidate], condition: list[str]) -> Candidate | None:
    """Highest selected-box score whose box covers a sensible image fraction; ties -> earlier."""
    lo, hi = frac_range(condition)
    best = None
    for c in cands:
        c.select_score, c.reject = None, ""
        det = c.detection
        if c.status != "ok":
            c.reject = c.status
            continue
        if not det:
            c.reject = "no detection"
            continue
        if not lo <= det["box_area_frac"] <= hi:
            c.reject = f"box covers {det['box_area_frac']:.0%} (want {lo:.0%}-{hi:.0%})"
            continue
        c.select_score = det["score"]
        if best is None or c.select_score > best.select_score:
            best = c
    return best


def detect_candidates(item_dir: Path, cands: list[Candidate], category: str, detector) -> None:
    import automask as am
    from PIL import Image, ImageOps

    targets, distractors = am.queries_for(category)
    for c in cands:
        if c.status != "ok":
            continue
        with Image.open(item_dir / "candidates" / c.file) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > 1600:  # OWLv2 works at 960 px; save CPU time
                im.thumbnail((1600, 1600))
            size = im.size
            dets = detector.detect(im, targets, distractors)
        d = am.select_box(dets, size)
        c.detection = (
            None
            if d is None
            else {
                **d.to_json(),
                "box_area_frac": round(am.box_fraction(d, size), 4),
                "image_size": list(size),
            }
        )
        print(
            f"    {c.file}: "
            + (f"{d.label} {d.score:.3f} ({c.detection['box_area_frac']:.0%})" if d else "-"),
            flush=True,
        )


# --------------------------------------------------------------------------- main


def download_candidates(fetcher: Fetcher, src: Source, item_dir: Path, rep: ItemReport):
    status, body = fetcher.page(src.url)
    reason = block_reason(status, body)
    if reason:
        rep.reason = reason
        return None, None
    cands = extract_candidates(body, src.url)[:MAX_CANDIDATES]
    cdir = item_dir / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "page.html").write_text(body)  # for re-parsing dims offline (--reselect)
    from PIL import Image

    for k, c in enumerate(cands):
        if not fetcher.allowed(c.url):
            c.status = "robots_disallowed"
            continue
        try:
            data = fetcher.image(c.url)
        except Exception as exc:
            c.status, c.reject = "download_failed", str(exc)[:200]
            continue
        ext = (IMG_EXT.search(c.url).group(1).lower() if IMG_EXT.search(c.url) else "jpg")
        c.file = f"{k:03d}.{ext.replace('jpeg', 'jpg')}"
        (cdir / c.file).write_bytes(data)
        try:
            with Image.open(cdir / c.file) as im:
                c.width, c.height = im.size
        except Exception:
            c.status = "not_image"
            (cdir / c.file).unlink()
            continue
        c.status = "ok" if min(c.width, c.height) >= MIN_SIDE else "too_small"
        if c.status != "ok":
            (cdir / c.file).unlink()
        print(f"    [{k + 1}/{len(cands)}] {c.status} {c.width}x{c.height} {c.url}", flush=True)
    return cands, body


def process(src: Source, data: Path, fetcher: Fetcher | None, detector, args) -> ItemReport:
    import make_mask

    rep = ItemReport(src.case, src.item_id, src.url)
    item_dir = data / src.item_id
    index = item_dir / "candidates" / "index.json"
    if not src.url:
        rep.reason = "no source URL (needs the owner's own photo, see photo_sources.md)"
        return rep
    if index.exists() and not args.refetch:
        cands = [Candidate(**c) for c in json.loads(index.read_text())]
        page_path = item_dir / "candidates" / "page.html"
        body = page_path.read_text() if page_path.exists() else ""
        rep.notes.append("reused downloaded candidates")
    else:
        if fetcher is None:
            rep.reason = "not downloaded yet (run without --reselect)"
            return rep
        if not fetcher.allowed(src.url):
            rep.reason = "robots.txt disallows the listing page"
            return rep
        try:
            cands, body = download_candidates(fetcher, src, item_dir, rep)
        except Exception as exc:
            rep.reason = f"page request failed: {type(exc).__name__}: {exc}"
            return rep
        if cands is None:
            return rep
    rep.n_candidates = sum(c.status == "ok" for c in cands)
    if not rep.n_candidates:
        rep.reason = "no usable candidate images on the page"
        _save_index(index, cands)
        return rep
    if detector is None:
        _save_index(index, cands)
        rep.reason = "downloaded; detection skipped (--no-detect)"
        return rep
    if args.redetect or any(c.status == "ok" and c.detection is None for c in cands):
        detect_candidates(item_dir, cands, src.category, detector)
    chosen = choose_candidate(cands, src.condition)
    _save_index(index, cands)
    if chosen is None:
        rep.reason = f"no candidate with a sensible '{src.category}' detection"
        return rep
    meta = {
        "category": src.category,
        "condition": src.condition,
        "notes": f"auto-selected by fetch_sources.py: candidate {chosen.file} of "
        f"{rep.n_candidates} ({chosen.origin}, '{chosen.detection['label']}' "
        f"{chosen.detection['score']:.3f}); EVAL case #{src.case}: {src.case_text}",
        "source_url": make_mask.strip_tracking(src.url),
        "image_url": chosen.url,
        "image_selection": {
            "detector": "google/owlv2-base-patch16-ensemble",
            "rule": "highest selected-box score with box fraction in "
            f"{frac_range(src.condition)}",
            "candidate_file": chosen.file,
            "detection": chosen.detection,
        },
    }
    dims = dims_from_page(body, src.url) if body else None
    if dims:
        meta["dims"], meta["dims_provenance"], rep.dims_text = dims[0], dims[1], dims[2]
        meta["dims_text"] = dims[2]
        rep.dims, rep.dims_provenance = dims[0], dims[1]
    else:
        rep.notes.append("no W/D/H stated on the page: dims unset (mesh scale kept)")
    old = make_mask._read_meta(item_dir)
    for k in ("true_footprint", "subtype", "prompt"):  # keep hand-added fields
        if k in old:
            meta[k] = old[k]
    image_changed = old.get("image_url") != chosen.url or not (item_dir / "image.jpg").exists()
    if image_changed:
        make_mask.prepare_item(item_dir / "candidates" / chosen.file, item_dir, meta)
    else:  # same photo: keep the existing mask and its record
        for k in ("mask_source", "auto_mask"):
            if k in old:
                meta[k] = old[k]
        make_mask._write_meta(item_dir, meta)
    rep.fetched = True
    rep.chosen_url = chosen.url
    rep.detector_score = chosen.detection["score"]
    rep.box_area_frac = chosen.detection["box_area_frac"]
    return rep


def _save_index(index: Path, cands: list[Candidate]) -> None:
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps([asdict(c) for c in cands], indent=1))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="fetch and auto-select the evaluation photos")
    p.add_argument("--data", type=Path, default=HERE / "data")
    p.add_argument("--sources", type=Path, default=HERE / "photo_sources.md")
    p.add_argument("--items", help="comma-separated case numbers (default: all)")
    p.add_argument("--reselect", action="store_true", help="no network; reuse candidates")
    p.add_argument("--refetch", action="store_true", help="download again even if present")
    p.add_argument("--redetect", action="store_true", help="re-run the detector")
    p.add_argument("--no-detect", action="store_true", help="download only")
    p.add_argument("--device", default="auto")
    p.add_argument("--delay", type=float, default=DELAY_S)
    args = p.parse_args(argv)
    sys.path.insert(0, str(HERE))

    sources = parse_sources(args.sources.read_text())
    if args.items:
        wanted = {int(x) for x in args.items.split(",")}
        sources = [s for s in sources if s.case in wanted]
    args.data.mkdir(parents=True, exist_ok=True)
    fetcher = None if args.reselect else Fetcher(args.delay)
    detector = None
    if not args.no_detect:
        import automask

        detector = automask.OwlDetector(device=args.device)
    reports = []
    for src in sources:
        print(f"[{src.item_id}] {src.url or '(no URL)'}", flush=True)
        rep = process(src, args.data, fetcher, detector, args)
        print(f"[{src.item_id}] fetched={rep.fetched} {rep.reason} {rep.chosen_url}", flush=True)
        reports.append(rep)
        try:  # blocked items leave no empty folder behind (run_sam3d would flag it)
            (args.data / src.item_id).rmdir()
        except OSError:
            pass
    report_path = args.data / "fetch_report.json"
    old = json.loads(report_path.read_text()) if report_path.exists() else {}
    old.update({str(r.case): asdict(r) for r in reports})
    report_path.write_text(json.dumps(old, indent=2))
    print("\n| # | item | fetched | reason | chosen image | dims (m) | provenance | score |")
    print("|---|---|---|---|---|---|---|---|")
    for r in reports:
        d = r.dims
        dims = f"{d['width']:.3f} x {d['depth']:.3f} x {d['height']:.3f}" if d else "-"
        print(
            f"| {r.case} | {r.item_id} | {'y' if r.fetched else 'n'} | {r.reason or '-'} | "
            f"{r.chosen_url or '-'} | {dims} | {r.dims_provenance or '-'} | "
            f"{r.detector_score if r.detector_score is not None else '-'} |"
        )
    print(f"\nreport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
