"""Preprocess FASTA + label files into a single labels TSV for causal saliency.

Reads:
  - data/vgp/all_vgp_tes.fa            (sequences; header lengths derived here)
  - data/vgp/features                  (header  TRUE/FALSE  -> binary TIR flag)
  - data/vgp/features-tpase            (header  superfamily string e.g. DNA/hAT)
  - the v4.3 checkpoint's superfamily list (only superfamilies the model knows
    about are kept; everything else is marked superfamily_in_model=0)

Writes a single TSV (no FASTA dependency at attribution time):

  header \\t genome \\t class \\t superfamily \\t superfamily_id \\t tir \\t
  seq_len \\t in_model_vocab \\t fa_offset \\t fa_byte_len

Columns:
  - class               : top-level (DNA / LTR / LINE / other) parsed from the
                          superfamily prefix
  - superfamily_id      : index into the model's superfamily_names list, or -1
  - in_model_vocab      : 1 if superfamily is in the v4.3 vocabulary
  - genome              : everything after the last '-' before '#' in header
  - fa_offset, fa_byte_len : byte offsets into all_vgp_tes.fa so the runner can
                          random-access a single record without rereading the
                          whole file. Sequence lines may include newlines, so
                          fa_byte_len covers the raw inclusive region from the
                          start of the '>' line to the byte before the next
                          '>' (or EOF).

Usage:
  ./.venv/bin/python model_result_interp/preprocess_labels.py \\
      --out model_result_interp/interpretation_results/causal_saliency_hybrid/labels.tsv

Notes on philosophy: this script is the ONLY place that does header parsing
and label joining. Downstream attribution code must read this TSV.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FASTA = REPO / "data" / "vgp" / "all_vgp_tes.fa"
DEFAULT_TIR = REPO / "data" / "vgp" / "features"
DEFAULT_SF = REPO / "data" / "vgp" / "features-tpase"
DEFAULT_CKPT = (REPO / "data_analysis" / "vgp_model_data_tpase_multi" / "v4.3"
                / "hybrid_v4.3_epoch40.pt")
DEFAULT_OUT = (REPO / "model_result_interp" / "interpretation_results"
               / "causal_saliency_hybrid" / "labels.tsv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    p.add_argument("--tir", type=Path, default=DEFAULT_TIR)
    p.add_argument("--sf", type=Path, default=DEFAULT_SF)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="v4.3 checkpoint; used only for the superfamily vocab.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def parse_class(sf: str) -> str:
    if sf.startswith("DNA"):
        return "DNA"
    if sf.startswith("LTR"):
        return "LTR"
    if sf.startswith("LINE"):
        return "LINE"
    if sf.startswith("SINE"):
        return "SINE"
    if sf.startswith("RC"):
        return "RC"  # rolling-circle / Helitron
    return "other"


def parse_genome(header: str) -> str:
    """Extract the genome accession (e.g. 'aAnoBae') from headers like
    'hAT_1-aAnoBae#DNA/hAT'.  Falls back to '' if pattern doesn't match.
    """
    base = header.split("#", 1)[0]
    if "-" in base:
        return base.rsplit("-", 1)[1]
    return ""


def load_two_col(path: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                continue
            h = parts[0].lstrip(">")
            d[h] = parts[1]
    return d


def stream_fasta_offsets(path: Path):
    """Yield (header, fa_offset, fa_byte_len, seq_len) for each record.
    Only one pass; lower-case characters are counted as sequence."""
    cur_h = None
    cur_offset = 0
    cur_seq_len = 0
    rec_start = 0
    with path.open("rb") as f:
        offset = 0
        for raw in f:
            line = raw  # bytes
            stripped = line.strip()
            if line.startswith(b">"):
                if cur_h is not None:
                    yield cur_h, rec_start, offset - rec_start, cur_seq_len
                cur_h = stripped[1:].decode("ascii", errors="ignore")
                rec_start = offset
                cur_seq_len = 0
            else:
                # count sequence bytes (excluding newline). uppercase or lowercase OK.
                cur_seq_len += sum(1 for b in stripped if 65 <= b <= 122)
            offset += len(line)
        if cur_h is not None:
            yield cur_h, rec_start, offset - rec_start, cur_seq_len


def load_sf_vocab_from_ckpt(path: Path) -> list[str]:
    if not path.exists():
        print(f"WARN: ckpt {path} not found; superfamily_id will all be -1.",
              file=sys.stderr)
        return []
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return list(ckpt.get("superfamily_names", []))


def main() -> int:
    args = parse_args()
    print(f"reading TIR labels from {args.tir} ...")
    tir = load_two_col(args.tir)
    print(f"  {len(tir)} entries")
    print(f"reading superfamily labels from {args.sf} ...")
    sfs = load_two_col(args.sf)
    print(f"  {len(sfs)} entries")

    print(f"reading sf vocab from {args.ckpt} ...")
    sf_vocab = load_sf_vocab_from_ckpt(args.ckpt)
    sf_to_id = {n: i for i, n in enumerate(sf_vocab)}
    print(f"  {len(sf_vocab)} superfamilies in model vocab")

    print(f"streaming fasta {args.fasta} ...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cnt_total = 0; cnt_full = 0; cnt_in_vocab = 0
    with args.out.open("w") as out:
        out.write("header\tgenome\tclass\tsuperfamily\tsuperfamily_id\ttir\t"
                  "seq_len\tin_model_vocab\tfa_offset\tfa_byte_len\n")
        for h, off, blen, slen in stream_fasta_offsets(args.fasta):
            cnt_total += 1
            sf = sfs.get(h, "")
            t = tir.get(h, "")
            tval = "1" if t.upper() in ("TRUE", "T", "1", "YES") else (
                   "0" if t.upper() in ("FALSE", "F", "0", "NO") else "")
            cls = parse_class(sf) if sf else ""
            sf_id = sf_to_id.get(sf, -1)
            in_vocab = "1" if sf in sf_to_id else "0"
            if sf and t:
                cnt_full += 1
            if in_vocab == "1":
                cnt_in_vocab += 1
            genome = parse_genome(h)
            out.write(f"{h}\t{genome}\t{cls}\t{sf}\t{sf_id}\t{tval}\t{slen}\t"
                      f"{in_vocab}\t{off}\t{blen}\n")
    print(f"wrote {args.out}")
    print(f"  total={cnt_total}  with_sf_and_tir={cnt_full}  in_model_vocab={cnt_in_vocab}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
