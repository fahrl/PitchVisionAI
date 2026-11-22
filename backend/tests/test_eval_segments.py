import sys
from collections import Counter
from pathlib import Path

import pandas as pd

P = Path(__file__).resolve().parents[2]
CLIP = sys.argv[1] if len(sys.argv) > 1 else "clip1"
SEG = 90            # frame per segmen (3 detik pada 30 fps)
MIN_VALID = 15      # segmen dengan frame valid kurang dari ini tidak dinilai

FORM_DIR = P / "outputs" / "formation_lines"
CSV_CLIP = FORM_DIR / f"formation_pitch_{CLIP}.csv"
CSV_IN = CSV_CLIP if CSV_CLIP.exists() else FORM_DIR / "formation_pitch.csv"
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
    f = pd.read_csv(CSV_IN)
    f["seg"] = (f.frame - 1) // SEG
    nseg = int(f.seg.max()) + 1

    if not LABELS.exists():
        rows = [{"clip": CLIP, "seg": s, "frames": f"{s * SEG + 1}-{(s + 1) * SEG}",
                 "team": t, "label_manual": ""}
                for s in range(nseg) for t in ("Team A", "Team B")]
        pd.DataFrame(rows).to_csv(LABELS, index=False)
        print("Template label dibuat:", LABELS)
        print("Isi kolom label_manual (format 3-4-3, jumlah 10 pemain luar, tanpa kiper)")
        print("SEBELUM melihat output algoritma. Kosongkan jika tidak yakin.")
        print("Lalu jalankan script ini lagi untuk menghitung kecocokan.")
        return

    lab = pd.read_csv(LABELS, dtype={"label_manual": str}).fillna("")
    out = []
    for r in lab.itertuples(index=False):
        g = f[(f.seg == r.seg) & (f.team == r.team)]
        algo = Counter(g.raw).most_common(1)[0][0] if len(g) else ""
        man, alg = parse(r.label_manual), parse(algo)
        valid = len(g)
        if man is None or alg is None or valid < MIN_VALID:
            status = "tidak dinilai"
            exact, diff = None, None
        else:
            exact = man == alg
            diff = sum(abs(a - b) for a, b in zip(man, alg))
            status = "cocok" if exact else "beda"
        out.append({"clip": r.clip, "seg": r.seg, "frames": r.frames, "team": r.team,
                    "label_manual": r.label_manual, "algoritma": algo,
                    "frame_valid": valid, "status": status, "selisih_pemain": diff})
    rep = pd.DataFrame(out)
    rep.to_csv(REPORT, index=False)

    print(rep.to_string(index=False))
    sc = rep[rep.status != "tidak dinilai"]
    print(f"\nSegmen dinilai: {len(sc)}/{len(rep)}")
    if len(sc):
        print(f"Cocok persis : {(sc.status == 'cocok').sum()}/{len(sc)}"
              f" ({(sc.status == 'cocok').mean() * 100:.0f}%)")
        print(f"Selisih rata-rata (pemain pindah baris): {sc.selisih_pemain.mean():.2f}")
    print("\nCSV:", REPORT)


if __name__ == "__main__":
    main()