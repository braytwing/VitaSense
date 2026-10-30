"""Demo en vivo / offline.

  python run_live.py                                  # webcam 0: FC + gráficas
  python run_live.py --rr                             # + FR (fusión; tórax por flujo óptico)
  python run_live.py --rr --alar                      # + aleteo nasal (EXPERIMENTAL, opt-in)
  python run_live.py --bp --bp-ref 118/76 --bp-save sesiones/s01.npz        # recolectar para PA
  python run_live.py --bp --bp-model bp_model.npz --bp-calib bp_calib.json  # PA calibrada
  python run_live.py --bp --bp-model bp_model.npz --bp-ref 118/76 --bp-save-calib bp_calib.json
  python run_live.py --video clip.mp4 --no-display --csv out.csv

Teclas: q = salir, r = reiniciar. Dos ventanas: cámara con ROIs y "Signos vitales" (onda de pulso,
evolución de la FC y de la FR). SpO2 sigue disponible con --spo2 (experimental, sin prioridad).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from rppg import (BloodPressureEstimator, BPCalibration, ChestMotionTracker, HeartRateEstimator,
                  PipelineConfig, RespiratoryRateEstimator, RidgeFeatureModel, RPPGPipeline,
                  SpO2Calibration, SpO2Estimator, save_session)
from rppg.roi import make_roi_extractor
from rppg.sources import VideoFileSource, WebcamSource
from rppg.viz import Dashboard

WHITE, GREEN, AMBER, ORANGE, BLUE = (220, 220, 220), (0, 255, 0), (0, 200, 255), (0, 120, 255), (255, 160, 0)


def vitals_lines(snap):
    """Líneas de texto (texto, color, escala) para el overlay."""
    lines = []
    hr = snap.hr
    txt = f"FC: {hr.value:.0f} lpm" if hr.value is not None else "FC: --"
    lines.append((txt, GREEN if snap.quality.score > 0.5 else AMBER, 1.0))
    lines.append((f"calidad {snap.quality.score:.2f}  SNR {(hr.extra.get('snr_db') if hr.extra.get('snr_db') is not None else float('nan')):.1f} dB", WHITE, 0.5))
    if snap.quality.flags:
        lines.append((" ".join(snap.quality.flags), ORANGE, 0.5))

    rr = snap.rr
    if rr is not None:
        if rr.value is not None:
            lines.append((f"FR: {rr.value:.0f} resp/min  (q {rr.quality:.2f})", GREEN if rr.quality > 0.5 else AMBER, 0.7))
        elif rr.extra.get("note") == "acumulando datos":
            lines.append((f"FR: acumulando datos (>{rr.extra['needs_s']:.0f} s)", WHITE, 0.6))
        else:
            lines.append(("FR: --", WHITE, 0.6))

    bp = snap.results.get("bp")
    if bp is not None:
        if bp.value is not None:
            lines.append((f"PA: {bp.extra['sbp']:.0f}/{bp.extra['dbp']:.0f} mmHg  (investigacion)", AMBER, 0.6))
        else:
            lines.append((f"PA: {bp.extra.get('note', '--')}", WHITE, 0.5))

    sp = snap.spo2
    if sp is not None:
        ex = sp.extra
        if sp.value is not None and ex.get("calibrated"):
            lines.append((f"SpO2: {sp.value:.0f} %  (q {sp.quality:.2f})", GREEN if sp.quality > 0.5 else AMBER, 0.7))
        elif sp.value is not None:
            lines.append((f"SpO2*: {sp.value:.0f} %  SIN CALIBRAR", ORANGE, 0.7))
        elif "ratio_rb" in ex:
            lines.append((f"SpO2: R={ex['ratio_rb']:.2f} (sin calibrar, no es %)", WHITE, 0.55))
    return lines


def draw_overlay(img, roi, snap, chest=None):
    if roi is not None:
        for poly in roi.polygons.values():
            cv2.polylines(img, [poly], True, GREEN, 1)
    if chest is not None:
        if chest.visible and chest.roi:
            x0, y0, x1, y1 = chest.roi
            cv2.rectangle(img, (x0, y0), (x1, y1), BLUE, 1)
        elif roi is not None:
            cv2.putText(img, "Torax fuera de cuadro: aleja la camara / muestra hombros",
                        (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)
    if snap is None or snap.hr is None:
        cv2.putText(img, "Calibrando...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        return
    y = 30
    for txt, color, scale in vitals_lines(snap):
        y += int(14 + 22 * scale)
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if scale >= 0.7 else 1)


def fmt(v, nd=1):
    return "" if v is None else f"{v:.{nd}f}"


def parse_ref(text):
    if not text:
        return None
    s, d = text.split("/")
    return float(s), float(d)


def save_timed_session(records, duration, dash, snap):
    """Exporta ventanas calculadas, resumen y grafica. No inventa mediciones ausentes."""
    folder = Path("measurements")
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder / f"remote_vital_{duration}s_{datetime.now():%Y%m%d_%H%M%S}"
    fields = ["t_s", "hr_lpm", "hr_raw_lpm", "hr_quality", "snr_db", "quality", "flags",
              "rr_resp_min", "rr_quality", "bp_sbp", "bp_dbp", "spo2_pct"]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    def summary(key, qkey, threshold):
        vals = [r[key] for r in records if r[key] is not None and
                r[qkey] is not None and r[qkey] >= threshold]
        return (float(np.median(vals)), len(vals)) if vals else (None, 0)
    hr, hn = summary("hr_lpm", "hr_quality", 0.3)
    rr, rn = summary("rr_resp_min", "rr_quality", 0.3)
    # Cobertura por actualizaciones, no porcentaje de frames validos.
    valid_hr = sum(1 for r in records if r["hr_lpm"] is not None and (r["hr_quality"] or 0) >= 0.3)
    valid_rr = sum(1 for r in records if r["rr_resp_min"] is not None and (r["rr_quality"] or 0) >= 0.3)
    lines = ["REMOTE VITAL - EXPERIMENTAL / NO DIAGNOSTICO",
             f"Duracion: {duration} s", f"Ventanas calculadas: {len(records)}",
             f"FC (mediana de ventanas aceptadas): {hr if hr is not None else 'SIN RESULTADO FIABLE'} lpm; ventanas={hn}",
             f"FR (mediana de ventanas aceptadas): {rr if rr is not None else 'SIN RESULTADO FIABLE'} resp/min; ventanas={rn}",
             f"Actualizaciones con FC aceptada: {valid_hr}/{len(records)} (NO es exactitud ni cobertura por frame)",
             f"Actualizaciones con FR aceptada: {valid_rr}/{len(records)} (NO es exactitud ni cobertura por frame)",
             "FR requiere al menos ~20 s de señal: una prueba de 15 s normalmente no produce FR.",
             "PA y SpO2 experimentales: no utilizar para decisiones medicas.",
             "Las ventanas se solapan y NO son observaciones independientes."]
    prefix.with_suffix(".txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    cv2.imwrite(str(prefix.with_name(prefix.name + "_graficas.png")), dash.render(snap))
    return prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--algo", default="pos", choices=["green", "chrom", "pos"])
    ap.add_argument("--roi", default="auto", choices=["auto", "mesh", "haar"])
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument(
        "--duration",
        type=int,
        choices=[15, 30, 60],
        default=None,
        help="Duracion de prueba; omitir para monitor continuo"
    )
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
    "--backend",
    choices=["auto", "dshow"],
    default="auto"
    )
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--no-dashboard", action="store_true", help="sin la ventana de gráficas")
    ap.add_argument("--csv", type=str, default=None)
    # respiración
    ap.add_argument("--rr", action="store_true", help="estimar frecuencia respiratoria")
    ap.add_argument("--no-chest", action="store_true", help="sin flujo óptico del tórax")
    ap.add_argument("--alar", action="store_true", help="añadir aleteo nasal como fuente (experimental)")
    # presión arterial (investigación)
    ap.add_argument("--bp", action="store_true", help="módulo de PA de investigación")
    ap.add_argument("--bp-model", type=str, default=None, help="modelo .npz (train_bp.py)")
    ap.add_argument("--bp-calib", type=str, default=None, help="calibración .json")
    ap.add_argument("--bp-ref", type=str, default=None, help="lectura de tu tensiómetro, p. ej. 118/76")
    ap.add_argument("--bp-save", type=str, default=None, help="guardar la sesión .npz para entrenar")
    ap.add_argument("--bp-save-calib", type=str, default=None, help="calcular y guardar el desfase por sujeto")
    # SpO2 (experimental)
    ap.add_argument("--spo2", action="store_true", help="razón R / SpO2 (experimental)")
    ap.add_argument("--spo2-calib", type=str, default=None)
    ap.add_argument("--spo2-uncalibrated", action="store_true")
    ap.add_argument("--spo2-pair", default="rb", choices=["rb", "rg"])
    args = ap.parse_args()
    if args.duration is not None:
        args.rr = True  # En sesiones temporizadas activar FR automaticamente.
    backend = cv2.CAP_DSHOW if args.backend == "dshow" else None
    source = (VideoFileSource(args.video) if args.video else WebcamSource(
        index=args.camera, width=args.width, height=args.height,
        fps=args.fps, backend=backend))
    roi = make_roi_extractor(args.roi)

    estimators = [HeartRateEstimator()]                 # la FC va primero: RR, PA y SpO2 la usan
    if args.rr:
        srcs = RespiratoryRateEstimator.ALL_SOURCES if args.alar else RespiratoryRateEstimator.SOURCES
        estimators.append(RespiratoryRateEstimator(sources=srcs))
    bp_est = None
    if args.bp:
        model = RidgeFeatureModel.load(args.bp_model) if args.bp_model else None
        cal = BPCalibration.load(args.bp_calib) if args.bp_calib else None
        bp_est = BloodPressureEstimator(model, cal, collect=bool(args.bp_save))
        estimators.append(bp_est)
    if args.spo2:
        scal = SpO2Calibration.load(args.spo2_calib) if args.spo2_calib else None
        estimators.append(SpO2Estimator(calibration=scal, pair=args.spo2_pair,
                                        allow_uncalibrated=args.spo2_uncalibrated))
    chest = ChestMotionTracker() if (args.rr and not args.no_chest) else None
    pipe = RPPGPipeline(roi, estimators, PipelineConfig(algorithm=args.algo, window_s=args.window), chest=chest)
    dash = Dashboard(show_rr=args.rr, show_quality=True)

    writer = fh = None
    if args.csv:
        fh = open(args.csv, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["t_s", "hr_lpm", "hr_raw_lpm", "snr_db", "quality", "flags", "rr_resp_min",
                         "rr_quality", "bp_sbp", "bp_dbp", "spo2_pct", "ratio_rb", "ratio_rg"])

    snap, dash_img = None, None
    duration = args.duration  # None = monitor continuo
    running = duration is None or args.no_display or bool(args.video)
    session_start = None
    records = []
    last_status = "Preparado. 1/2/3: duracion; ESPACIO: iniciar; R: reiniciar; Q: salir"
    saved_session = None
    try:
        with source:
            while True:
                frame = source.read()
                if frame is None:
                    break
                # La duracion de una prueba se mide por reloj monotono, no por numero de frames.
                now = frame.t
                if running and session_start is None:
                    session_start = now
                    pipe.reset()
                    dash.reset()
                    snap, dash_img = None, None
                    records = []
                    last_status = "Midiendo..."
                if running:
                    s = pipe.process_frame(frame)
                    if s is not None:
                        snap = s
                        dash.update(s)
                        hr, rr, sp, bp = s.hr, s.rr, s.spo2, s.results.get("bp")
                        elapsed = max(0.0, now - session_start)
                        row = {
                            "t_s": round(elapsed, 3),
                            "hr_lpm": hr.value if hr else None,
                            "hr_raw_lpm": hr.extra.get("raw_bpm") if hr else None,
                            "hr_quality": hr.quality if hr else None,
                            "snr_db": hr.extra.get("snr_db") if hr else None,
                            "quality": s.quality.score,
                            "flags": "|".join(s.quality.flags),
                            "rr_resp_min": rr.value if rr else None,
                            "rr_quality": rr.quality if rr else None,
                            "bp_sbp": bp.extra.get("sbp") if bp else None,
                            "bp_dbp": bp.extra.get("dbp") if bp else None,
                            "spo2_pct": sp.value if sp else None,
                        }
                        records.append(row)
                        if writer:
                            # Formato del CSV continuo original, con tiempo absoluto del pipeline.
                            writer.writerow([
                                f"{s.t:.2f}", fmt(hr.value), fmt(hr.extra.get("raw_bpm")),
                                fmt(hr.extra.get("snr_db"), 2), f"{s.quality.score:.2f}",
                                "|".join(s.quality.flags), fmt(rr.value) if rr else "",
                                fmt(rr.quality, 2) if rr else "", fmt(bp.extra.get("sbp")) if bp else "",
                                fmt(bp.extra.get("dbp")) if bp else "", fmt(sp.value) if sp else "",
                                fmt(sp.extra.get("ratio_rb"), 4) if sp else "",
                                fmt(sp.extra.get("ratio_rg"), 4) if sp else ""
                            ])
                        if args.no_display:
                            print(f"t={elapsed:5.1f}s FC={fmt(hr.value)} FR={fmt(rr.value) if rr else '-'} calidad={s.quality.score:.2f}")
                if running and duration is not None and now - session_start >= duration:
                    running = False
                    saved_session = save_timed_session(records, duration, dash, snap)
                    last_status = f"Finalizado. Resultados en {saved_session}. ESPACIO: repetir"
                    print(last_status)
                if not args.no_display:
                    draw_overlay(frame.image, pipe.last_roi, snap, chest)
                    if duration is not None:
                        elapsed = min(duration, max(0, now - session_start)) if session_start is not None else 0
                        banner = f"{duration}s | {elapsed:.0f}/{duration}s | {'GRABANDO' if running else 'ESPERA/FINALIZADO'}"
                        cv2.putText(frame.image, banner, (10, frame.image.shape[0]-42),
                                    cv2.FONT_HERSHEY_SIMPLEX, .55, AMBER, 2)
                    cv2.putText(frame.image, "1=15s 2=30s 3=60s ESPACIO=iniciar R=reset Q=salir",
                                (10, frame.image.shape[0]-18), cv2.FONT_HERSHEY_SIMPLEX, .42, WHITE, 1)
                    cv2.imshow("rPPG", frame.image)
                    if not args.no_dashboard:
                        # Congelar la ultima grafica al finalizar la sesion.
                        if running or dash_img is None:
                            dash_img = dash.render(snap)
                        cv2.imshow("Signos vitales", dash_img)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 27):
                        break
                    if k in (ord("1"), ord("2"), ord("3")):
                        duration = {ord("1"): 15, ord("2"): 30, ord("3"): 60}[k]
                        running = False
                        session_start = None
                        pipe.reset(); dash.reset(); snap = dash_img = None
                        records = []
                        print(f"Seleccionados {duration} segundos. Pulsa ESPACIO para comenzar.")
                    elif k == ord(" "):
                        if duration is not None:
                            running = True
                            session_start = None
                    elif k == ord("r"):
                        pipe.reset(); dash.reset()
                        snap = dash_img = None
                        session_start = None
                        records = []
                        running = duration is None
                elif duration is not None and not running:
                    break
    finally:
        roi.close()
        if fh:
            fh.close()
        cv2.destroyAllWindows()

    ref = parse_ref(args.bp_ref)
    if bp_est is not None:
        if args.bp_save and bp_est.records:
            save_session(args.bp_save, bp_est.records, ref=ref)
            print(f"Sesión de PA guardada en {args.bp_save} ({len(bp_est.records)} ventanas, ref={ref})")
        if args.bp_save_calib and ref and bp_est.raw_preds:
            cal = BPCalibration.fit(np.mean(bp_est.raw_preds, axis=0)[None], np.array([ref]))
            cal.save(args.bp_save_calib)
            print(f"Calibración guardada en {args.bp_save_calib}: SBP {cal.sbp_offset:+.1f}  DBP {cal.dbp_offset:+.1f}")


if __name__ == "__main__":
    main()
