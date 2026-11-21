"""Organise ASL Citizen (Microsoft Research) for extract_landmarks, for a chosen set of classes.

    # 1. see how the classes match, how many videos and how big - downloads only ~15 MB of metadata
    python -m scripts.prepare_asl_citizen --dry-run
    # 2. fetch + link the videos, then extract landmarks
    python -m scripts.prepare_asl_citizen --map delay=POSTPONE
    python -m data_collection.extract_landmarks --src datasets/ASL_CITIZEN --out datasets/asl_citizen_landmarks --workers 4

--src can be
  * the official URL (default): the 46 GB zip is NOT downloaded; its table of contents is
    read with HTTP range requests and only the needed videos are fetched (~0.55 MB each)
    into --raw. Re-running skips videos that are already there.
  * a local ASL_Citizen.zip    : the needed videos are extracted into --raw
  * an unzipped folder         : containing splits/*.csv and videos/ - videos are only linked

Output: --out/<gloss>/<video>.mp4 symlinks (lowercase gloss names, matching the class
names in --classes-from, i.e. the WLASL classes), plus --out/splits.json with the
official signer-independent split of every video ("<gloss>/<stem>": train|val|test),
which extract_landmarks copies next to the landmarks and training respects.

Matching: an ASL Citizen gloss matches a class if it is equal after lowercasing and
removing punctuation and a trailing variant number (BED1, BED2 -> bed). --variants first
keeps only the lowest-numbered variant. --map class=GLOSS[+GLOSS2] overrides a class.

License: ASL Citizen is for non-commercial research only and may not be redistributed.
Keep the videos out of git (datasets/ is ignored).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import struct
import sys
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.config import DATASETS_DIR, LANDMARK_DIR

OFFICIAL_URL = ("https://download.microsoft.com/download/b/8/8/"
                "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip")
SPLITS = ("train", "val", "test")


# ------------------------------------------------------------ remote zip
class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests (enough for zipfile's table of contents)."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos, self._cache = url, block, 0, {}
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        if head.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise SystemExit("Server does not support range requests; download the zip and pass --src ZIP")
        self.size = int(head.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def get(self, start: int, end: int) -> bytes:
        """Bytes [start, end) straight from the server."""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == end - start:
                    return data
            except OSError:
                if attempt == 3:
                    raise
        raise OSError(f"short read for bytes {start}-{end}")

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n >= self.block:  # big read: no caching
            data = self.get(self.pos, self.pos + n)
            self.pos += n
            return data
        out = bytearray()
        while n > 0:
            bi = self.pos // self.block
            if bi not in self._cache:
                self._cache = {bi: self.get(bi * self.block, min(self.size, (bi + 1) * self.block))}
            chunk = self._cache[bi][self.pos - bi * self.block:][:n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


def fetch_member(raw: HTTPRangeFile, info: zipfile.ZipInfo) -> bytes:
    """Download and decompress one zip member with a single range request."""
    slack = 30 + len(info.filename.encode()) + 1024  # local header + name + generous extra field
    blob = raw.get(info.header_offset, min(raw.size, info.header_offset + slack + info.compress_size))
    sig, *_, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", blob[:30])
    if sig != 0x04034B50:
        raise ValueError(f"bad local header for {info.filename}")
    start = 30 + name_len + extra_len
    data = blob[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(data) != info.CRC:
        raise ValueError(f"CRC mismatch for {info.filename}")
    return data


# ---------------------------------------------------------------- sources
class Source:
    """Uniform access to the split CSVs and videos of a URL, zip or folder."""

    def __init__(self, src: str):
        self.folder = self.zip = self.remote = None
        p = Path(src).expanduser()
        if src.startswith(("http://", "https://")):
            self.remote = HTTPRangeFile(src)
            self.zip = zipfile.ZipFile(self.remote)
        elif p.is_dir():
            self.folder = p if (p / "splits").is_dir() else p / "ASL_Citizen"
            if not (self.folder / "splits").is_dir():
                raise SystemExit(f"No splits/ folder in {p}")
        elif p.is_file():
            self.zip = zipfile.ZipFile(p)
        else:
            raise SystemExit(f"--src not found: {src}")
        if self.zip:
            self.members = {Path(i.filename).name: i for i in self.zip.infolist()
                            if not i.is_dir()}

    def split_rows(self, split: str) -> list[dict]:
        if self.folder:
            text = (self.folder / "splits" / f"{split}.csv").read_text(encoding="utf-8-sig")
        else:
            text = self.zip.read(self.members[f"{split}.csv"]).decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(text)))

    def video_size(self, name: str) -> int:
        if self.folder:
            return (self.folder / "videos" / name).stat().st_size
        return self.members[name].compress_size

    def has_video(self, name: str) -> bool:
        return (self.folder / "videos" / name).exists() if self.folder else name in self.members

    def materialise(self, name: str, raw_dir: Path) -> Path:
        """Path of a local copy of the video (fetched/extracted into raw_dir if needed)."""
        if self.folder:
            return (self.folder / "videos" / name).resolve()
        dst = raw_dir / name
        if dst.exists():
            return dst.resolve()
        info = self.members[name]
        data = fetch_member(self.remote, info) if self.remote else self.zip.read(info)
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dst)
        return dst.resolve()


# --------------------------------------------------------------- matching
def norm(gloss: str) -> str:
    return re.sub(r"[^a-z0-9]", "", gloss.lower())


def base_and_variant(gloss: str) -> tuple[str, int]:
    m = re.match(r"^(.*?)(\d+)$", gloss)
    return (norm(m.group(1)), int(m.group(2))) if m else (norm(gloss), 0)


def match_glosses(classes: list[str], glosses: set[str], overrides: dict[str, list[str]],
                  variants: str) -> dict[str, list[str]]:
    by_base: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for g in glosses:
        base, var = base_and_variant(g)
        by_base[base].append((var, g))
    out = {}
    for c in classes:
        if c in overrides:
            missing = [g for g in overrides[c] if g not in glosses]
            if missing:
                raise SystemExit(f"--map {c}: unknown ASL Citizen gloss(es) {missing}")
            out[c] = overrides[c]
            continue
        found = sorted(by_base.get(norm(c), []))
        if variants == "first" and found:
            found = found[:1]
        out[c] = [g for _, g in found]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=OFFICIAL_URL, help="URL, ASL_Citizen.zip or unzipped folder")
    ap.add_argument("--out", type=Path, default=DATASETS_DIR / "ASL_CITIZEN")
    ap.add_argument("--raw", type=Path, default=DATASETS_DIR / "asl_citizen_raw",
                    help="where fetched/extracted videos are stored (URL/zip sources)")
    ap.add_argument("--classes", nargs="*", help="class names (default: folders in --classes-from)")
    ap.add_argument("--classes-from", type=Path, default=LANDMARK_DIR,
                    help="use the class folders of this landmark dir (default: the WLASL classes)")
    ap.add_argument("--map", nargs="*", default=[], metavar="CLASS=GLOSS[+GLOSS]",
                    help="manual matches, e.g. delay=POSTPONE")
    ap.add_argument("--variants", choices=["all", "first"], default="all",
                    help="numbered sign variants (BED1, BED2): use all, or only the first")
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads")
    ap.add_argument("--dry-run", action="store_true", help="only show matches, counts and size")
    args = ap.parse_args()

    classes = args.classes or sorted(p.name for p in args.classes_from.iterdir() if p.is_dir())
    overrides = {}
    for item in args.map:
        c, _, gl = item.partition("=")
        if not gl:
            raise SystemExit(f"--map expects class=GLOSS, got {item!r}")
        overrides[c.strip().lower()] = [g.strip() for g in gl.split("+") if g.strip()]

    print("Reading ASL Citizen metadata from", args.src)
    src = Source(args.src)
    rows = [(split, r) for split in SPLITS for r in src.split_rows(split)]
    glosses = {r["Gloss"] for _, r in rows}
    matches = match_glosses(classes, glosses, overrides, args.variants)

    gloss_to_class = {g: c for c, gs in matches.items() for g in gs}
    selected = [(split, r) for split, r in rows if r["Gloss"] in gloss_to_class]

    print(f"\n{len(classes)} classes, {len(glosses)} ASL Citizen glosses\n")
    print(f"{'class':<14} {'ASL Citizen gloss(es)':<38} {'train':>5} {'val':>4} {'test':>4}")
    per = Counter((gloss_to_class[r['Gloss']], s) for s, r in selected)
    for c in classes:
        gs = "+".join(matches[c]) or "-- NO MATCH --"
        print(f"{c:<14} {gs:<38} {per[c, 'train']:>5} {per[c, 'val']:>4} {per[c, 'test']:>4}")
    unmatched = [c for c in classes if not matches[c]]
    present = [r for _, r in selected if src.has_video(r["Video file"])]
    size = sum(src.video_size(r["Video file"]) for r in present)
    print(f"\nunmatched: {unmatched or 'none'}")
    print(f"{len(selected)} videos selected ({len(selected) - len(present)} missing from the archive), "
          f"{size / 1e9:.2f} GB")
    if args.dry_run:
        return

    if src.folder is None:
        args.raw.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(args.raw).free
        if free < size * 1.2:
            raise SystemExit(f"Not enough disk space: need ~{size / 1e9:.1f} GB, {free / 1e9:.1f} GB free")

    def job(item):
        split, r = item
        name = r["Video file"]
        cls = gloss_to_class[r["Gloss"]]
        dst = args.out / cls / name
        key = f"{cls}/{Path(name).stem}"
        if dst.is_symlink() or dst.exists():
            return key, split, "already there"
        try:
            target = src.materialise(name, args.raw)
        except Exception as e:  # noqa: BLE001 - report and continue
            return key, split, f"error: {e}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(target)
        return key, split, "linked"

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_):
            return x

    stats: Counter = Counter()
    split_of: dict[str, str] = {}
    workers = args.workers if src.remote else 1  # local zipfile objects are not thread-safe
    with ThreadPoolExecutor(workers) as pool:
        for key, split, status in tqdm(pool.map(job, [x for x in selected if src.has_video(x[1]["Video file"])]),
                                       total=len(present)):
            stats[status.split(":")[0]] += 1
            if status.startswith("error"):
                print(key, status, file=sys.stderr)
            else:
                split_of[key] = split

    splits_path = args.out / "splits.json"
    old = json.loads(splits_path.read_text()) if splits_path.exists() else {}
    splits_path.write_text(json.dumps({**old, **split_of}, indent=0, sort_keys=True))
    print(dict(stats))
    print(f"{len(split_of)} videos linked in {args.out}, splits in {splits_path}")
    print("Next: python -m data_collection.extract_landmarks --src", args.out,
          "--out", DATASETS_DIR / "asl_citizen_landmarks", "--workers 4")


if __name__ == "__main__":
    main()
