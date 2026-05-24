"""
Face Tracker v2 — Enhanced AI HUD
===================================
• Face detection   : OpenCV Haar Cascade
• Age + Gender     : InsightFace buffalo_s  (ONNX, auto-downloads ~75 MB once)
• Emotion          : MediaPipe FaceLandmarker blendshapes  (52 ARKit coefficients)
• Head pose        : Euler angles from MediaPipe facial transformation matrix
• Eye openness     : Per-eye blink blendshapes
• Timeline         : Scrolling emotion history strip
• Distance         : Estimated from face-width / frame-width ratio

Press Q to quit.
"""

import cv2
import numpy as np
import threading
import time
import os
import urllib.request
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FONT   = cv2.FONT_HERSHEY_DUPLEX    # cleaner than SIMPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX

# Palette (BGR)
ACCENT   = (0,   210, 160)
WHITE    = (245, 245, 245)
DIM      = (115, 120, 130)
BLACK    = (0,   0,   0)
PANEL_BG = (12,  12,  18)

EMOTION_COLORS = {
    "HAPPY":     (0,   215,  80),
    "SAD":       (205,  65,   0),
    "ANGRY":     (0,    20, 225),
    "SURPRISED": (0,   175, 255),
    "FEAR":      (165,   0, 165),
    "DISGUSTED": (0,   115,  40),
    "NEUTRAL":   (175, 175, 175),
}
EMOTIONS = ["HAPPY", "SAD", "ANGRY", "SURPRISED", "FEAR", "DISGUSTED", "NEUTRAL"]

# Landmark indices used for face-mesh dot overlay (MediaPipe 478-point topology)
_MESH_INDICES = list(dict.fromkeys([
    # Left / right eye contours
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    # Eyebrows
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
    336, 296, 334, 293, 300, 383, 353, 276, 283, 282, 295, 285,
    # Lips outer + inner
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95,
    # Nose
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 326,
    # Face oval (every other point)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]))

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe model – auto-download
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_DIR  = os.path.expanduser("~/.face_tracker")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")
_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")


def _ensure_model() -> bool:
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if os.path.exists(_MODEL_PATH):
        return True
    print("[INFO] Downloading face landmark model (~4 MB)…")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[INFO] Model ready.")
        return True
    except Exception:
        try:
            import requests as _r
            resp = _r.get(_MODEL_URL, timeout=30)
            resp.raise_for_status()
            with open(_MODEL_PATH, "wb") as f:
                f.write(resp.content)
            print("[INFO] Model ready.")
            return True
        except Exception as e:
            print(f"[WARN] Could not download landmark model: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy imports
# ─────────────────────────────────────────────────────────────────────────────
_MP_OK = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mpp
    from mediapipe.tasks.python import vision as _mpv
    _MP_OK = True
except Exception as e:
    print(f"[WARN] MediaPipe unavailable: {e}")

_IF_OK = False
try:
    from insightface.app import FaceAnalysis as _IFA
    _IF_OK = True
except Exception as e:
    print(f"[WARN] InsightFace unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Blendshape → emotion (52 ARKit coefficients from MediaPipe)
# ─────────────────────────────────────────────────────────────────────────────
def _bs_dict(blendshapes) -> dict:
    """Convert mediapipe blendshape list to {name: score} dict."""
    out = {}
    if blendshapes is None:
        return out
    for item in blendshapes:
        name  = getattr(item, "category_name", None) or getattr(item, "label", "")
        score = getattr(item, "score", 0.0)
        out[name] = float(score)
    return out


def blendshapes_to_scores(blendshapes) -> dict:
    bs = _bs_dict(blendshapes)

    def avg(*keys):
        vals = [bs.get(k, 0) for k in keys]
        return sum(vals) / max(len(vals), 1)

    happy     = avg("mouthSmileLeft",     "mouthSmileRight",
                    "cheekSquintLeft",    "cheekSquintRight")
    sad       = avg("mouthFrownLeft",     "mouthFrownRight") * 0.5 \
              + bs.get("browInnerUp", 0) * 0.3 \
              + avg("mouthLowerDownLeft", "mouthLowerDownRight") * 0.2
    angry     = avg("browDownLeft",  "browDownRight") * 0.55 \
              + avg("eyeSquintLeft", "eyeSquintRight") * 0.25 \
              + avg("mouthFrownLeft","mouthFrownRight") * 0.20
    surprised = bs.get("jawOpen", 0) * 0.45 \
              + avg("eyeWideLeft",       "eyeWideRight") * 0.35 \
              + avg("browOuterUpLeft",   "browOuterUpRight") * 0.20
    fear      = avg("eyeWideLeft",       "eyeWideRight") * 0.35 \
              + bs.get("browInnerUp", 0) * 0.30 \
              + bs.get("jawOpen", 0) * 0.20 \
              + avg("mouthStretchLeft",  "mouthStretchRight") * 0.15
    disgusted = avg("noseSneerLeft",     "noseSneerRight") * 0.55 \
              + avg("mouthFrownLeft",    "mouthFrownRight") * 0.30 \
              + avg("browDownLeft",      "browDownRight")   * 0.15

    raw = {"HAPPY": happy, "SAD": sad, "ANGRY": angry,
           "SURPRISED": surprised, "FEAR": fear, "DISGUSTED": disgusted}
    top     = max(raw.values(), default=0)
    neutral = max(0.0, 1.0 - top * 1.8)
    raw["NEUTRAL"] = neutral

    total = sum(raw.values()) + 1e-9
    return {k: float(v / total * 100) for k, v in raw.items()}


def eye_open_pct(blendshapes) -> tuple:
    bs = _bs_dict(blendshapes)
    return (1 - bs.get("eyeBlinkLeft",  0)) * 100, \
           (1 - bs.get("eyeBlinkRight", 0)) * 100


def smile_pct(blendshapes) -> float:
    bs = _bs_dict(blendshapes)
    return (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2 * 100


def matrix_to_euler(mat) -> tuple:
    """Yaw, pitch, roll in degrees from a 4×4 row-major matrix."""
    try:
        R = np.array(mat, dtype=float).reshape(4, 4)[:3, :3]
        pitch = float(np.degrees(np.arctan2(-R[2, 0],
                                            np.sqrt(R[2, 1]**2 + R[2, 2]**2))))
        yaw   = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        roll  = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        return yaw, pitch, roll
    except Exception:
        return 0.0, 0.0, 0.0


def est_distance(face_w: int, frame_w: int) -> int:
    """Rough cm estimate — empirical for a typical laptop webcam."""
    ratio = face_w / max(frame_w, 1)
    return int(0.25 * 65 / max(ratio, 0.01))


# ─────────────────────────────────────────────────────────────────────────────
# Face state — temporal smoothing + timeline
# ─────────────────────────────────────────────────────────────────────────────
class FaceState:
    def __init__(self):
        self.scores   = {e: 100 / 7 for e in EMOTIONS}
        self.timeline = deque(maxlen=180)   # ~6 s at 30 fps

    def smooth(self, new_scores: dict, alpha=0.28):
        for k in EMOTIONS:
            self.scores[k] = alpha * new_scores.get(k, 0) \
                           + (1 - alpha) * self.scores.get(k, 0)
        self.timeline.append(max(self.scores, key=self.scores.__getitem__))

    @property
    def dominant(self) -> str:
        return max(self.scores, key=self.scores.__getitem__)


# ─────────────────────────────────────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────────────────────────────────────
def _txt(img, text, x, y, color=WHITE, scale=0.58, thick=1):
    """Outlined text — readable on any background."""
    cv2.putText(img, text, (x+1, y+1), FONT, scale, BLACK, thick+2, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), FONT, scale, color, thick,   cv2.LINE_AA)


def _stxt(img, text, x, y, color=DIM, scale=0.42, thick=1):
    """Small outlined text using SIMPLEX."""
    cv2.putText(img, text, (x+1, y+1), FONT_S, scale, BLACK, thick+2, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), FONT_S, scale, color, thick,   cv2.LINE_AA)


def _panel(img, x, y, w, h, alpha=0.82, border=None):
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x+w, y+h), PANEL_BG, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x+w, y+h), border or (52, 58, 72), 1)


def _divider(img, px, py, pw, color=(52, 58, 72)):
    cv2.line(img, (px+4, py), (px+pw-4, py), color, 1)


def _hbar(img, x, y, w, h, pct, color, bg=(32, 32, 42)):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    fill = max(1, int(w * pct / 100))
    cv2.rectangle(img, (x, y), (x+fill, y+h), color, -1)


def _corners(img, x, y, w, h, color, arm, thick=3, anim=0.0):
    a = int(arm + anim * 7)
    segs = [
        ((x,   y),   (x+a, y)),   ((x,   y),   (x,   y+a)),
        ((x+w, y),   (x+w-a, y)), ((x+w, y),   (x+w, y+a)),
        ((x,   y+h), (x+a, y+h)), ((x,   y+h), (x,   y+h-a)),
        ((x+w, y+h), (x+w-a, y+h)), ((x+w, y+h), (x+w, y+h-a)),
    ]
    for p1, p2 in segs:
        cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)


def _scan_line(img, x, y, w, h, frame_n, color, period=55):
    pos = int((frame_n % period) / period * h)
    sy  = y + pos
    if y <= sy <= y + h:
        ov = img.copy()
        cv2.line(ov, (x, sy), (x+w, sy), color, 1)
        cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)


def _face_mesh(img, landmarks, W, H, alpha=0.50):
    ov = img.copy()
    for idx in _MESH_INDICES:
        if idx < len(landmarks):
            lm = landmarks[idx]
            cv2.circle(ov, (int(lm.x * W), int(lm.y * H)),
                       1, (0, 195, 155), -1, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _emotion_bars(img, scores, ox, oy, bw=108, bh=13, gap=6):
    for i, emo in enumerate(EMOTIONS):
        by  = oy + i * (bh + gap)
        col = EMOTION_COLORS[emo]
        pct = scores.get(emo, 0)
        _hbar(img, ox, by, bw, bh, pct, col)
        _stxt(img, f"{emo[:7]:<7} {pct:4.0f}%",
              ox + bw + 5, by + bh - 1, col, scale=0.38)


def _pose_compass(img, yaw, pitch, cx, cy, r=28):
    """Mini compass: dot position encodes yaw + pitch."""
    cv2.circle(img, (cx, cy), r, (45, 50, 60), 1, cv2.LINE_AA)
    cv2.line(img, (cx-r, cy), (cx+r, cy), (35, 40, 50), 1)
    cv2.line(img, (cx, cy-r), (cx, cy+r), (35, 40, 50), 1)
    dx = int(np.clip(np.sin(np.radians(yaw)),   -1, 1) * r * 0.9)
    dy = int(np.clip(np.sin(np.radians(pitch)), -1, 1) * r * 0.9)
    cv2.circle(img, (cx + dx, cy + dy), 5, (180, 220, 255), -1, cv2.LINE_AA)
    cv2.line(img,  (cx, cy), (cx+dx, cy+dy), (120, 160, 200), 1, cv2.LINE_AA)


def _timeline(img, history, x, y, W, bar_h=14):
    tl = list(history)
    n  = len(tl)
    _panel(img, x, y, W, bar_h + 22, alpha=0.80, border=(45, 50, 62))
    _stxt(img, "EMOTION  TIMELINE", x+6, y+13, DIM, scale=0.38)
    if n == 0:
        return
    bw = max(1, W // max(n, 1))
    for i, emo in enumerate(tl):
        bx = x + i * bw
        cv2.rectangle(img, (bx, y+18), (bx+bw, y+18+bar_h),
                      EMOTION_COLORS.get(emo, DIM), -1)


# ─────────────────────────────────────────────────────────────────────────────
# InsightFace age / gender (lazy-loaded background thread)
# ─────────────────────────────────────────────────────────────────────────────
class AgeGender:
    def __init__(self):
        self._app   = None
        self._ready = False
        self._lock  = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        if not _IF_OK:
            return
        try:
            print("[INFO] Loading InsightFace buffalo_s (first run: ~75 MB)…")
            a = _IFA(name="buffalo_s", providers=["CPUExecutionProvider"])
            a.prepare(ctx_id=0, det_size=(320, 320))
            with self._lock:
                self._app, self._ready = a, True
            print("[INFO] InsightFace ready.")
        except Exception as e:
            print(f"[WARN] InsightFace: {e}")

    def analyse(self, bgr) -> list:
        with self._lock:
            if not self._ready:
                return []
            app = self._app
        try:
            return [{"age": int(f.age), "gender": f.sex or "?"}
                    for f in app.get(bgr)]
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe FaceLandmarker (Tasks API, mediapipe ≥ 0.10)
# ─────────────────────────────────────────────────────────────────────────────
class Landmarker:
    def __init__(self):
        self._lmk   = None
        self._ready = False
        self._lock  = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        if not _MP_OK or not _ensure_model():
            return
        try:
            opts = _mpv.FaceLandmarkerOptions(
                base_options=_mpp.BaseOptions(model_asset_path=_MODEL_PATH),
                num_faces=4,
                min_face_detection_confidence=0.45,
                min_face_presence_confidence=0.45,
                min_tracking_confidence=0.45,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            lmk = _mpv.FaceLandmarker.create_from_options(opts)
            with self._lock:
                self._lmk, self._ready = lmk, True
            print("[INFO] MediaPipe FaceLandmarker ready.")
        except Exception as e:
            print(f"[WARN] FaceLandmarker: {e}")

    def detect(self, rgb_img) -> list:
        """Returns [{landmarks, blendshapes, matrix}, …]"""
        with self._lock:
            if not self._ready:
                return []
            lmk = self._lmk
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            r      = lmk.detect(mp_img)
            out    = []
            for i, fl in enumerate(r.face_landmarks or []):
                # fl may be a list or a protobuf with .landmark
                lms = list(fl.landmark) if hasattr(fl, "landmark") else list(fl)
                bs  = None
                if r.face_blendshapes and i < len(r.face_blendshapes):
                    bs_raw = r.face_blendshapes[i]
                    bs = (list(bs_raw.classification)
                          if hasattr(bs_raw, "classification") else list(bs_raw))
                mat = None
                if r.facial_transformation_matrixes and \
                   i < len(r.facial_transformation_matrixes):
                    mat = r.facial_transformation_matrixes[i]
                out.append({"landmarks": lms, "blendshapes": bs, "matrix": mat})
            return out
        except Exception:
            return []

    def close(self):
        with self._lock:
            if self._lmk:
                try:
                    self._lmk.close()
                except Exception:
                    pass
                self._lmk = None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cascade    = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    ag_det     = AgeGender()
    landmarker = Landmarker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    # Wait up to 8 s for camera permission on macOS
    deadline = time.time() + 8
    while time.time() < deadline:
        ok, _ = cap.read()
        if ok:
            break
        print("[INFO] Waiting for camera… "
              "(grant access: System Settings > Privacy & Security > Camera)")
        time.sleep(1)
    else:
        print("\nERROR: Cannot open camera.")
        print("Grant camera access in System Settings > Privacy & Security > Camera")
        cap.release()
        return

    fps_q  = deque(maxlen=30)
    prev_t = time.time()
    frame_n = 0

    # Age/gender background cache
    ag_cache: dict = {}
    ag_lock  = threading.Lock()
    ag_busy  = False

    def _ag_worker(img, ids):
        nonlocal ag_busy
        for i, r in enumerate(ag_det.analyse(img)):
            if i < len(ids):
                with ag_lock:
                    ag_cache[ids[i]] = r
        ag_busy = False

    face_states: dict[str, FaceState] = {}

    print("=" * 65)
    print("  Face Tracker v2  |  Mood · Age · Gender · Pose · Timeline")
    print("  Press  Q  to quit.")
    print("=" * 65)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        now = time.time()
        fps_q.append(1.0 / max(now - prev_t, 1e-9))
        prev_t  = now
        frame_n += 1
        fps     = sum(fps_q) / len(fps_q)
        anim    = (np.sin(frame_n * 0.10) + 1) / 2   # [0, 1] pulse

        H, W   = frame.shape[:2]
        display = frame.copy()
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Detection ────────────────────────────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=5, minSize=(70, 70))
        lm_faces = landmarker.detect(rgb)

        # Age/gender every 30 frames
        if frame_n % 30 == 0 and not ag_busy and len(boxes) > 0:
            ag_busy = True
            ids = [f"f{i}" for i in range(len(boxes))]
            threading.Thread(target=_ag_worker,
                             args=(frame.copy(), ids), daemon=True).start()

        # Purge dead face states
        active = {f"f{i}" for i in range(len(boxes))}
        for k in list(face_states):
            if k not in active:
                del face_states[k]

        # ── Per-face render ───────────────────────────────────────────────────
        for i, (x, y, w, h) in enumerate(boxes):
            fid    = f"f{i}"
            cx, cy = x + w // 2, y + h // 2
            if fid not in face_states:
                face_states[fid] = FaceState()
            state = face_states[fid]

            # Match closest landmark face by nose-tip proximity
            best, best_d = None, float("inf")
            for lfd in lm_faces:
                lms = lfd["landmarks"]
                nx  = lms[4].x * W
                ny  = lms[4].y * H
                d   = abs(nx - cx) + abs(ny - cy)
                if d < best_d:
                    best_d, best = d, lfd

            landmarks   = None
            blendshapes = None
            matrix      = None
            eye_l, eye_r = 80.0, 80.0
            sm_pct       = 0.0
            yaw, pitch, roll = 0.0, 0.0, 0.0

            if best and best_d < w * 1.5:
                landmarks   = best["landmarks"]
                blendshapes = best["blendshapes"]
                matrix      = best["matrix"]
                if blendshapes:
                    state.smooth(blendshapes_to_scores(blendshapes))
                    eye_l, eye_r = eye_open_pct(blendshapes)
                    sm_pct       = smile_pct(blendshapes)
                if matrix is not None:
                    yaw, pitch, roll = matrix_to_euler(matrix)

            scores   = dict(state.scores)
            dominant = state.dominant
            color    = EMOTION_COLORS.get(dominant, DIM)
            dist     = est_distance(w, W)

            with ag_lock:
                ag = ag_cache.get(fid, {})
            age    = ag.get("age",    None)
            gender = ag.get("gender", None)

            # ── Visuals ───────────────────────────────────────────────────────
            if landmarks:
                _face_mesh(display, landmarks, W, H, alpha=0.55)

            _scan_line(display, x, y, w, h, frame_n, color)

            # Tint inside face box
            ov = display.copy()
            cv2.rectangle(ov, (x, y), (x+w, y+h), color, -1)
            cv2.addWeighted(ov, 0.07, display, 0.93, 0, display)

            cv2.rectangle(display, (x, y), (x+w, y+h), (38, 42, 52), 1)
            _corners(display, x, y, w, h, color, arm=30, thick=3, anim=anim)

            # Face-centre crosshair
            cv2.drawMarker(display, (cx, cy), color,
                           cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)

            # Head-pose compass (above the face)
            compass_cy = max(38, y - 42)
            _pose_compass(display, yaw, pitch, cx, compass_cy, r=28)

            # ── Info panel ────────────────────────────────────────────────────
            pw, ph = 208, 248
            px = x - pw - 10 if x - pw - 10 > 0 else x + w + 10
            py = max(40, min(y, H - ph - 6))
            _panel(display, px, py, pw, ph, border=color)

            lp  = px + 10
            row = py + 20

            # ── Face ID header ────────────────────────────────────────────────
            _txt(display, f"  FACE  # {i+1}", lp - 4, row, color, scale=0.62)
            row += 6
            _divider(display, px, row, pw, color)
            row += 14

            # ── Mood ──────────────────────────────────────────────────────────
            _stxt(display, "MOOD",   lp,      row, DIM, scale=0.44)
            _txt( display, dominant, lp + 58, row, color, scale=0.56)
            row += 20
            # Confidence bar
            conf = scores.get(dominant, 0)
            _hbar(display, lp, row, pw - 20, 7, conf, color)
            _stxt(display, f"{conf:.0f}%", lp + pw - 18, row + 6,
                  color, scale=0.38)
            row += 18

            # ── Smile intensity ───────────────────────────────────────────────
            _stxt(display, "SMILE", lp, row, DIM, scale=0.40)
            _hbar(display, lp + 52, row - 6, 80, 7, sm_pct,
                  EMOTION_COLORS["HAPPY"])
            _stxt(display, f"{sm_pct:.0f}%", lp + 138, row,
                  EMOTION_COLORS["HAPPY"], scale=0.38)
            row += 14
            _divider(display, px, row, pw)
            row += 12

            # ── Age & Gender ──────────────────────────────────────────────────
            _stxt(display, "AGE",    lp,      row, DIM, scale=0.44)
            _txt( display, f"~{age} yrs" if age else "loading…",
                  lp + 58, row, WHITE, scale=0.52)
            row += 22
            _stxt(display, "GENDER", lp,      row, DIM, scale=0.44)
            if gender == "M":
                g_col, g_val = (80, 190, 255), "Male"
            elif gender == "F":
                g_col, g_val = (255, 150, 210), "Female"
            else:
                g_col, g_val = DIM, "loading…"
            _txt(display, g_val, lp + 58, row, g_col, scale=0.52)
            row += 22
            _divider(display, px, row, pw)
            row += 12

            # ── Head pose ─────────────────────────────────────────────────────
            _stxt(display, "YAW",   lp,       row, DIM,   scale=0.42)
            _stxt(display, f"{yaw:+.0f}°",
                  lp + 40,  row, WHITE, scale=0.46)
            _stxt(display, "PITCH", lp + 88,  row, DIM,   scale=0.42)
            _stxt(display, f"{pitch:+.0f}°",
                  lp + 138, row, WHITE, scale=0.46)
            row += 18
            _stxt(display, "ROLL",  lp,       row, DIM,   scale=0.42)
            _stxt(display, f"{roll:+.0f}°",
                  lp + 40,  row, WHITE, scale=0.46)
            _stxt(display, "DIST",  lp + 88,  row, DIM,   scale=0.42)
            _stxt(display, f"~{dist} cm",
                  lp + 138, row, WHITE, scale=0.46)
            row += 18
            _divider(display, px, row, pw)
            row += 12

            # ── Eye openness bars ─────────────────────────────────────────────
            _stxt(display, "L.EYE", lp, row, DIM, scale=0.40)
            _hbar(display, lp + 52, row - 6, 72, 7, eye_l, (0, 200, 220))
            _stxt(display, f"{eye_l:.0f}%", lp + 130, row,
                  (0, 200, 220), scale=0.38)
            row += 16
            _stxt(display, "R.EYE", lp, row, DIM, scale=0.40)
            _hbar(display, lp + 52, row - 6, 72, 7, eye_r, (0, 200, 220))
            _stxt(display, f"{eye_r:.0f}%", lp + 130, row,
                  (0, 200, 220), scale=0.38)
            row += 16
            _divider(display, px, row, pw)
            row += 10

            # ── Position ──────────────────────────────────────────────────────
            _stxt(display, f"POS  ({cx}, {cy})   SIZE {w}×{h}",
                  lp, row, DIM, scale=0.38)

            # ── Emotion bars (right of face) ───────────────────────────────────
            bx = x + w + 12
            if bx + 240 < W:
                bpanel_h = len(EMOTIONS) * (13 + 6) + 10
                _panel(display, bx - 6, y - 4, 240, bpanel_h,
                       border=(52, 58, 72))
                _emotion_bars(display, scores, bx, y, bw=108)

        # ── Emotion timeline ──────────────────────────────────────────────────
        if face_states:
            _timeline(display, next(iter(face_states.values())).timeline,
                      0, H - 42, W)

        # ── Top HUD bar ───────────────────────────────────────────────────────
        _panel(display, 0, 0, W, 36, alpha=0.90, border=(42, 48, 60))
        _txt(display,
             f"  FACE TRACKER v2"
             f"     FACES: {len(boxes)}"
             f"     FPS: {fps:4.1f}"
             f"     FRAME: {frame_n:05d}"
             f"     [ Q ]  QUIT",
             8, 24, ACCENT, scale=0.60, thick=1)

        # Faint centre crosshair on empty frame
        cv2.line(display, (W//2-14, H//2), (W//2+14, H//2), (48, 52, 62), 1)
        cv2.line(display, (W//2, H//2-14), (W//2, H//2+14), (48, 52, 62), 1)

        cv2.imshow("Face Tracker v2  —  Mood · Age · Pose", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Tracker stopped.")


if __name__ == "__main__":
    main()
