"""Ajusta SpO2 = a - b*R a partir de CSV generados con run_live.py --spo2 --spo2-ref <lectura>.

  python calibrate_spo2.py sesion1.csv sesion2.csv --pair rb --out calib.json

Cada CSV debe tener las columnas ratio_rb / ratio_rg y spo2_ref (la lectura de tu oxímetro).
Imprime el ajuste; si el rango de referencia es estrecho (p. ej. solo 97-99 %) avisa de que
el resultado no es fiable. No provoques desaturaciones para ampliar el rango.
"""
import argparse
import csv

from rppg.spo2 import fit_calibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--pair", default="rb", choices=["rb", "rg"])
    ap.add_argument("--out", default="calib.json")
    args = ap.parse_args()

    ratios, refs = [], []
    for path in args.csv:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                r, y = row.get(f"ratio_{args.pair}", ""), row.get("spo2_ref", "")
                if r and y:
                    ratios.append(float(r))
                    refs.append(float(y))
    cal = fit_calibration(ratios, refs, pair=args.pair)
    cal.save(args.out)
    print(f"n={cal.n}  SpO2 = {cal.a:.2f} - {cal.b:.2f}·R   R²={cal.r2:.3f}  RMSE={cal.rmse:.2f} puntos")
    print(f"rango de referencia: {cal.ref_min:.0f}-{cal.ref_max:.0f} %  ->  guardado en {args.out}")


if __name__ == "__main__":
    main()
