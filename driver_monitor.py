"""Driver Fatigue & Distraction Monitor
MediaPipe FaceLandmarker / Face Mesh (TensorFlow Lite models, Tasks API) + OpenCV.
Keys: m = mesh on/off | r = recalibrate | q = quit | l = log on/off | s = sound on/off
      0/1/2 = log label (0 normal, 1 distracted, 2 fatigue)
Logging: python driver_monitor.py --log --label 0   (see train_fatigue_model.py to train the ML model)
"""
import argparse
import csv
import time
from collections import deque

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from alerts_audio import AlertPlayer
from features import FEATURES

# ---- Tunable parameters ---------------------------------------------------
CALIB_SEC = 3.0                              # baseline capture time
DISTRACT_SEC = 5.0                           # sustained head deviation -> alert
YAW_T, PITCH_T, ROLL_T = 30.0, 25.0, 30.0   # degrees away from baseline
EAR_RATIO = 0.75                             # closed if EAR < 0.75 * baseline EAR
YAWN_DELTA, YAWN_HOLD = 0.40, 1.2            # yawn: MAR > base+0.40 for 1.2 s
MICROSLEEP_SEC = 1.5                         # continuous eye closure -> critical
WINDOW = 60.0                                # PERCLOS / blink-rate window (s)
PERCLOS_FULL = 0.30                          # PERCLOS at which its meter term is maxed (lower = more sensitive)
ML_SMOOTH = 0.08                             # per-frame smoothing of the ML probabilities
ML_FATIGUE_ALERT = 60                        # fatigue score (0-100) that triggers the alert
ML_DISTRACT_P = 0.60                         # P(distracted) above which the ML counts as "away"

# ---- Labels used for logging / ML (class index = label) -------------------
LABELS = ["normal", "distracted", "fatigue"]

# ---- Landmarks (MediaPipe Face Mesh indices) --------------------------------
R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_V = [(82, 87), (13, 14), (312, 317)]
MOUTH_H = (78, 308)
PNP_IDS = [1, 152, 33, 263, 61, 291]         # nose, chin, eye corners, mouth corners
MODEL_3D = np.array([[0, 0, 0], [0, 63.6, 12.5], [-43.3, -32.7, 26.0],
                     [43.3, -32.7, 26.0], [-28.9, 28.9, 24.1], [28.9, 28.9, 24.1]],
                    dtype=np.float64)        # generic face, camera axes (y down, z away)

GREEN, YELLOW, RED, WHITE = (0, 200, 0), (0, 200, 255), (0, 0, 255), (255, 255, 255)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "face_landmarker.task")
LOG_PATH = os.path.join(HERE, "data", "features.csv")
FATIGUE_MODEL = os.path.join(HERE, "fatigue_model.keras")
TICK = 0.2                                   # s between feature rows / ML predictions


def load_connections():
    """Mesh edges from the Tasks API; empty lists -> draw dots instead."""
    from mediapipe.tasks.python.vision import face_landmarker as fl
    for src in (getattr(vision, "FaceLandmarksConnections", None), fl):
        tess = getattr(src, "FACE_LANDMARKS_TESSELATION", None)
        if tess:
            cont = getattr(src, "FACE_LANDMARKS_CONTOURS", [])
            return [(c.start, c.end) for c in tess], [(c.start, c.end) for c in cont]
    return [], []


def draw_mesh(img, p, tess, cont):
    pts = p.astype(int).tolist()
    if not tess:
        for x, y in pts:
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        return
    for a, b in tess:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (180, 180, 180), 1)
    for a, b in cont:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 1)


def dist(p, a, b):
    return float(np.linalg.norm(p[a] - p[b]))


def eye_ar(p, i):
    return (dist(p, i[1], i[5]) + dist(p, i[2], i[4])) / (2 * dist(p, i[0], i[3]))


def mouth_ar(p):
    return sum(dist(p, a, b) for a, b in MOUTH_V) / (3 * dist(p, *MOUTH_H))


def head_pose(p, w, h):
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, p[PNP_IDS], cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return np.array(cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0])  # pitch, yaw, roll


def text(img, s, org, color=WHITE, scale=0.55, th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def open_log():
    """Open data/features.csv for appending (header written only for a new/empty file)."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    is_new = not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0
    f = open(LOG_PATH, "a", newline="")
    w = csv.writer(f)
    if is_new:
        w.writerow(["session", "label"] + FEATURES)
    return f, w


def load_fatigue_model():
    """3-class Keras model (normal / distracted / fatigue) trained by train_fatigue_model.py,
    or None -> rule-based meter."""
    if not os.path.exists(FATIGUE_MODEL):
        return None
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        model = tf.keras.models.load_model(FATIGUE_MODEL)
        if model.output_shape[-1] != len(LABELS):
            print("fatigue_model.keras is an old 2-class model - delete it and retrain. "
                  "Using rule-based meter.")
            return None
        print("Loaded fatigue_model.keras - meter now uses the ML score.")
        return model
    except Exception as e:
        print("Could not load fatigue model, using rule-based meter:", e)
        return None


def new_state():
    return dict(base=None, calib=[], t_calib=None, t_start=None, closed_since=None,
                hist=deque(), blinks=deque(), yawns=deque(), yawn_since=None,
                yawn_counted=False, away_since=None, fatigue=0.0,
                ml=np.array([1.0, 0.0, 0.0]),   # smoothed P(normal, distracted, fatigue)
                pr=np.array([1.0, 0.0, 0.0]),   # latest raw model output
                last_tick=0.0, sess=str(int(time.time() * 1000)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="video file instead of the webcam")
    ap.add_argument("--log", action="store_true", help="record feature rows to data/features.csv")
    ap.add_argument("--label", type=int, default=0, choices=[0, 1, 2],
                    help="0 = normal, 1 = distracted, 2 = fatigue")
    args = ap.parse_args()
    ui = {"mesh": True, "log": args.log, "label": args.label, "rows": 0, "sound": True}
    BTN = (10, 10, 190, 42)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and BTN[0] <= x <= BTN[2] and BTN[1] <= y <= BTN[3]:
            ui["mesh"] = not ui["mesh"]

    if not os.path.exists(MODEL_PATH):
        print("Downloading face_landmarker.task (~4 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    tess, cont = load_connections()
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_faces=1))
    t0 = 0.0 if args.video else time.time()
    last_ts, nf = -1, 0
    model = load_fatigue_model()
    logf = logw = None                       # opened lazily on the first logged row
    player = AlertPlayer()
    cap = cv2.VideoCapture(args.video or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cv2.namedWindow("Driver Monitor")
    cv2.setMouseCallback("Driver Monitor", on_mouse)
    S = new_state()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        nf += 1
        h, w = frame.shape[:2]
        now = nf / fps if args.video else time.time()
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        last_ts = max(last_ts + 1, int((now - t0) * 1000))
        res = landmarker.detect_for_video(mp_img, last_ts)
        lm = res.face_landmarks[0] if res.face_landmarks else None
        ear = mar = pose = None
        if lm:
            p = np.array([[l.x * w, l.y * h] for l in lm])
            if ui["mesh"]:
                draw_mesh(frame, p, tess, cont)
            ear = (eye_ar(p, R_EYE) + eye_ar(p, L_EYE)) / 2
            mar = mouth_ar(p)
            pose = head_pose(p, w, h)
        face_ok = lm is not None and pose is not None

        ov = frame.copy()
        cv2.rectangle(ov, (0, 50), (500, 185), (0, 0, 0), -1)
        frame = cv2.addWeighted(ov, 0.5, frame, 0.5, 0)
        alerts, base = [], S["base"]

        if base is None:  # ---- Phase 1: baseline calibration (first 3 s)
            if face_ok:
                S["t_calib"] = S["t_calib"] or now
                S["calib"].append([ear, mar, *pose])
                prog = min((now - S["t_calib"]) / CALIB_SEC, 1.0)
                if prog >= 1.0:
                    v = np.median(S["calib"], axis=0)
                    S["base"], S["t_start"] = dict(ear=v[0], mar=v[1], pose=v[2:]), now
                text(frame, f"CALIBRATING {prog * 100:.0f}% - look straight ahead,",
                     (10, 75), YELLOW)
                text(frame, "eyes open, mouth closed", (10, 97), YELLOW)
            else:
                S["t_calib"], S["calib"] = None, []
                text(frame, "No face detected - face the camera", (10, 75), RED)
        else:  # ---- Phase 2: monitoring
            away, dev, cdur = not face_ok, np.zeros(3), 0.0
            if face_ok:
                dev = (pose - base["pose"] + 180) % 360 - 180   # pitch, yaw, roll deviation
                away = abs(dev[0]) > PITCH_T or abs(dev[1]) > YAW_T or abs(dev[2]) > ROLL_T
                closed = ear < EAR_RATIO * base["ear"]
                if closed:
                    S["closed_since"] = S["closed_since"] or now
                elif S["closed_since"]:
                    if now - S["closed_since"] < 1.0:          # short closure = blink
                        S["blinks"].append(now)
                    S["closed_since"] = None
                cdur = now - S["closed_since"] if S["closed_since"] else 0.0
                S["hist"].append((now, closed))
                if mar > base["mar"] + YAWN_DELTA:
                    S["yawn_since"] = S["yawn_since"] or now
                    if not S["yawn_counted"] and now - S["yawn_since"] >= YAWN_HOLD:
                        S["yawns"].append(now)
                        S["yawn_counted"] = True
                else:
                    S["yawn_since"], S["yawn_counted"] = None, False

            while S["hist"] and now - S["hist"][0][0] > WINDOW:
                S["hist"].popleft()
            while S["blinks"] and now - S["blinks"][0] > WINDOW:
                S["blinks"].popleft()
            while S["yawns"] and now - S["yawns"][0] > 180:
                S["yawns"].popleft()

            n = len(S["hist"])
            ramp = min(1.0, (now - S["t_start"]) / 20)          # warm-up for PERCLOS
            perclos = ramp * sum(c for _, c in S["hist"]) / n if n else 0.0
            bpm = len(S["blinks"]) * 60 / max(10.0, min(WINDOW, now - S["t_start"]))
            target = 100 * (0.50 * min(perclos / PERCLOS_FULL, 1) + 0.25 * min(cdur / 2.0, 1)
                            + 0.25 * min(len(S["yawns"]) / 3, 1))
            f = S["fatigue"]
            S["fatigue"] = f = f + (0.15 if target > f else 0.01) * (target - f)  # fast up, slow down

            # ---- feature tick: ML prediction + optional logging
            if face_ok and now - S["last_tick"] >= TICK:
                S["last_tick"] = now
                feat = [ear / base["ear"], mar - base["mar"], perclos, bpm, len(S["yawns"]), cdur,
                        abs(dev[0]), abs(dev[1]), abs(dev[2])]
                if model is not None:
                    S["pr"] = np.array(model(np.array([feat], dtype=np.float32),
                                             training=False)[0])
                if ui["log"]:
                    if logw is None:
                        logf, logw = open_log()
                    logw.writerow([S["sess"], ui["label"]] + [round(float(x), 4) for x in feat])
                    logf.flush()
                    ui["rows"] += 1

            # ---- ML probabilities, smoothed every frame
            if model is not None:
                S["ml"] += ML_SMOOTH * (S["pr"] - S["ml"])
                f = 100 * S["ml"][2]                            # P(fatigue) drives the meter

            # ---- distraction: head pose rule OR ML says "distracted"
            ml_away = model is not None and S["ml"][1] > ML_DISTRACT_P
            away_flag = away or ml_away
            S["away_since"] = (S["away_since"] or now) if away_flag else None
            away_sec = now - S["away_since"] if S["away_since"] else 0.0

            if cdur >= MICROSLEEP_SEC:
                alerts.append("MICROSLEEP - WAKE UP!")
            elif f >= ML_FATIGUE_ALERT:
                alerts.append("FATIGUE ALERT - TAKE A BREAK")
            if away_sec >= DISTRACT_SEC:
                alerts.append("DISTRACTION - EYES ON THE ROAD")

            if not face_ok:
                facing = "FACE LOST"
            elif not away:
                facing = "FORWARD (ML: distracted)" if ml_away else "FORWARD"
            else:
                facing = " ".join(filter(None, [
                    "RIGHT" if dev[1] > YAW_T else "LEFT" if dev[1] < -YAW_T else "",
                    "DOWN" if dev[0] > PITCH_T else "UP" if dev[0] < -PITCH_T else "",
                    "TILT-L" if dev[2] > ROLL_T else "TILT-R" if dev[2] < -ROLL_T else ""]))
            if face_ok:
                rows = [f"EAR {ear:.2f}  (base {base['ear']:.2f}, closed < {EAR_RATIO * base['ear']:.2f})",
                        f"MAR {mar:.2f}  (base {base['mar']:.2f}, yawn > {base['mar'] + YAWN_DELTA:.2f})",
                        f"Pitch {dev[0]:+.0f}  Yaw {dev[1]:+.0f}  Roll {dev[2]:+.0f} deg vs baseline",
                        f"PERCLOS {perclos * 100:.0f}%  Blinks/min {bpm:.0f}  Yawns(3m) {len(S['yawns'])}"]
            else:
                rows = ["No face detected"]
            rows.append(f"Facing: {facing}   Away {away_sec:.1f}/{DISTRACT_SEC:.0f}s")
            for i, r in enumerate(rows):
                text(frame, r, (10, 75 + 22 * i), RED if (i == len(rows) - 1 and away_flag) else WHITE)

            col = GREEN if f < 30 else YELLOW if f < 60 else RED
            cv2.rectangle(frame, (10, h - 40), (310, h - 20), WHITE, 1)
            cv2.rectangle(frame, (10, h - 40), (10 + int(3 * f), h - 20), col, -1)
            text(frame, f"FATIGUE {f:.0f}/100 ({'ML' if model is not None else 'rule'})",
                 (320, h - 24), col, 0.6, 2)
            if model is not None:
                text(frame, f"Normal {S['ml'][0]:.2f}  Distracted {S['ml'][1]:.2f}  "
                            f"Fatigue {S['ml'][2]:.2f}", (10, h - 50), WHITE, 0.5)

        if ui["sound"]:
            player.update({"fatigue" if a.startswith(("MICROSLEEP", "FATIGUE")) else "distraction"
                           for a in alerts})
        for i, a in enumerate(alerts):
            y = h - 70 - 45 * i
            cv2.rectangle(frame, (0, y - 32), (w, y + 8), RED, -1)
            text(frame, a, (15, y), WHITE, 0.9, 2)

        cv2.rectangle(frame, BTN[:2], BTN[2:], (60, 60, 60), -1)
        text(frame, f"Mesh: {'ON' if ui['mesh'] else 'OFF'} (m)", (BTN[0] + 10, BTN[1] + 22))
        if ui["log"]:
            text(frame, f"REC {ui['label']}={LABELS[ui['label']]} rows={ui['rows']}",
                 (200, 32), RED, 0.6, 2)
        cv2.imshow("Driver Monitor", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("m"):
            ui["mesh"] = not ui["mesh"]
        if k == ord("r"):
            S = new_state()
        if k in (ord("0"), ord("1"), ord("2")):
            ui["label"] = k - ord("0")
        if k == ord("l"):
            ui["log"] = not ui["log"]
        if k == ord("s"):
            ui["sound"] = not ui["sound"]

    landmarker.close()
    if logf:
        logf.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
