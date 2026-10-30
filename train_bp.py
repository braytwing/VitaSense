"""Entrena y EVALÚA un modelo de PA con tus sesiones etiquetadas (leave-one-session-out).

Flujo:
  1) por cada sesión, mide tu PA con un tensiómetro validado y graba:
       python run_live.py --bp --bp-ref 118/76 --bp-save sesiones/s01.npz
  2) repite en momentos/condiciones distintos (>= 10 sesiones es un mínimo razonable)
  3) python train_bp.py sesiones/*.npz --out bp_model.npz

La salida compara el modelo con el predictor trivial "media del entrenamiento". Si el modelo no
lo supera con claridad, NO ha aprendido nada útil, por mucho que el gráfico "se vea bien".
"""
import argparse

from rppg.bp import cross_validate_sessions, fit_all, load_session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--alpha", type=float, default=1.0, help="regularización ridge")
    ap.add_argument("--out", default="bp_model.npz")
    args = ap.parse_args()

    sessions = [load_session(p) for p in args.sessions]
    res = cross_validate_sessions(sessions, args.alpha)

    print(f"\nSesiones válidas: {res['n_sessions']}   (leave-one-session-out)")
    print(f"DE de la PA de referencia entre sesiones: SBP {res['ref_sd']['sbp']:.1f}  DBP {res['ref_sd']['dbp']:.1f} mmHg")
    if min(res["ref_sd"].values()) < 8:
        print("AVISO: la PA de referencia casi no varía entre sesiones; el predictor trivial es casi "
              "imbatible y cualquier mejora aparente puede ser azar.")
    for nm in ("sbp", "dbp"):
        r = res[nm]
        m, b = r["model"], r["baseline"]
        print(f"\n{nm.upper()}  modelo:   ME {m['me']:+.1f} ± {m['sd']:.1f}  MAE {m['mae']:.1f}  r {m['r']:.2f}")
        print(f"     trivial:  ME {b['me']:+.1f} ± {b['sd']:.1f}  MAE {b['mae']:.1f}")
        print(f"     ¿supera al trivial? {'SI' if r['beats_baseline'] else 'NO'}   "
              f"| criterio de referencia 5 ± 8 mmHg: {'cumple' if m['within_5_8'] else 'NO cumple'}")
    print("\nCon pocas sesiones estas cifras son muy inciertas; el criterio 5 ± 8 mmHg (ISO 81060-2) "
          "exige decenas de sujetos y es solo una referencia aquí.")

    fit_all(sessions, args.alpha).save(args.out)
    print(f"\nModelo entrenado con todas las sesiones guardado en {args.out}")


if __name__ == "__main__":
    main()
