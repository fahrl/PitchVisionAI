import sys
from collections import Counter

import pandas as pd

from clip_config import CLIP, P, out

if len(sys.argv) > 1 and sys.argv[1] != CLIP:
    raise SystemExit(f"Argumen '{sys.argv[1]}' beda dengan CLIP='{CLIP}'. Jalankan: set CLIP={sys.argv[1]}")

SEG = 90
MIN_VALID = 15

CSV_IN = out("formation_lines") / "formation_pitch.csv"
EVAL_DIR = P / "outputs" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
LABELS = EVAL_DIR / f"segment_labels_{CLIP}.csv"
REPORT = EVAL_DIR / f"segment_report_{CLIP}.csv"


def parse(s):
    try:
        v = [int(x) for x in str(s).strip().split("-")]
        return v if len(v) == 3 else None
    except ValueError:
        return None


def main():
    print("CLIP:", CLIP)
    f = pd.read_csv(CSV_IN)
    f["seg"] = (f.frame - 1) // SEG
    nseg = int(f.seg.max()) + 1

    if not LABELS.exists():
        rows = [{"clip": CLIP, "seg": s, "frames": f"{s * SEG + 1}-{(s + 1) * SEG}",
                 "team": t, "label_manual": ""}
                for s in range(nseg) for t in ("Team A", "Team B")]
        pd.DataFrame(rows).to_csv(LABELS, index=False)
        print("Template label dibuat:", LABELS)
        print("Isi label_manual SEBELUM melihat output, lalu jalankan lagi.")
        return

    lab = pd.read_csv(LABELS, dtype={"label_manual": str, "frames": str}).fillna("")
    res = []
    for r in lab.itertuples(index=False):
        g = f[(f.seg == r.seg) & (f.team == r.team)]
        algo = Counter(g.raw).most_common(1)[0][0] if len(g) else ""
        man, alg = parse(r.label_manual), parse(algo)
        if man is None or alg is None or len(g) < MIN_VALID:
            status, diff = "tidak dinilai", None
        else:
            diff = sum(abs(a - b) for a, b in zip(man, alg))
            status = "cocok" if man == alg else "beda"
        res.append({"clip": r.clip, "seg": r.seg, "frames": r.frames, "team": r.team,
                    "label_manual": r.label_manual, "algoritma": algo,
                    "frame_valid": len(g), "status": status, "selisih_pemain": diff})
    rep = pd.DataFrame(res)
    rep.to_csv(REPORT, index=False)
    print(rep.to_string(index=False))
    sc = rep[rep.status != "tidak dinilai"]
    print(f"\nSegmen dinilai: {len(sc)}/{len(rep)}")
    if len(sc):
        print(f"Cocok persis : {(sc.status == 'cocok').sum()}/{len(sc)} ({(sc.status == 'cocok').mean() * 100:.0f}%)")
        print(f"Selisih rata-rata: {sc.selisih_pemain.mean():.2f}")
    print("CSV:", REPORT)


if __name__ == "__main__":
    main()