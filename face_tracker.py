"""
Face Tracker — Mood, Age & Gender Detector
--------------------------------------------
• Face detection  : OpenCV Haar Cascade (bundled with cv2)
• Age + Gender    : InsightFace buffalo_s (ONNX, ~75 MB, auto-downloads once)
• Emotion         : MediaPipe FaceLandmarker Tasks API + geometric analysis
                    (model ~2 MB, auto-downloads to ~/.face_tracker/)
• Display         : OpenCV

Press Q to quit.
"""

import cv2
import numpy as np
import threading
import time
import os
import urllib.request
from collections import deque

# ── MediaPipe landmark model (auto-download) ──────────────────────────────────
_MODEL_DIR  = os.path.expanduser("~/.face_tracker")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_landmark_model():
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if not os.path.exists(_MODEL_PATH):
        print("[INFO] Downloading face landmark model (~4 MB)...")
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        except Exception:
            # Fallback: use requests (handles macOS SSL cert issues)
            try:
                import requests as _req
                r = _req.get(_MODEL_URL, timeout=30)
                r.raise_for_status()
                with open(_MODEL_PATH, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"[WARN] Could not download landmark model: {e}")
                return False
        print("[INFO] Landmark model ready.")
    return True


# ── Optional heavy imports (degrade gracefully) ───────────────────────────────
_MP_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _MP_AVAILABLE = True
except Exception as e:
    print(f"[WARN] MediaPipe unavailable: {e} — emotion detection disabled")

_IF_AVAILABLE = False
try:
    from insightface.app import FaceAnalysis as _IFApp
    _IF_AVAILABLE = True
except Exception as e:
    print(f"[WARN] InsightFace unavailable: {e} — age/gender disabled")


# ── Emotion colours (BGR) ─────────────────────────────────────────────────────
EMOTION_COLORS = {
    "HAPPY":     (0,   215,  80),
    "SAD":       (210,  70,   0),
    "ANGRY":     (0,    25, 220),
    "SURPRISED": (0,   175, 255),
    "FEAR":      (170,   0, 170),
    "DISGUSTED": (0,   120,  45),
    "NEUTRAL":   (195, 195, 195),
}
DEFAULT_COLOR = (195, 195, 195)

# MediaPipe FaceLandmark indices used for geometric emotion
# (478-point mesh from FaceLandmarker — standard MediaPipe topology)
_L_EYE  = [33,  160, 158, 133, 153, 144]  # EAR left
_R_EYE  = [362, 385, 387, 263, 373, 380]  # EAR right
_L_BROW = [70,  63,  105,  66, 107]
_R_BROW = [300, 293, 334,  296, 336]
_MOUTH_CORNER_L = 61
_MOUTH_CORNER_R = 291
_MOUTH_TOP      = 13
_MOUTH_BOT      = 14
_NOSE_TIP       = 4


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _pt(lm_list, idx, W, H):
    p = lm_list[idx]
    return np.array([p.x * W, p.y * H])


def _ear(lm, indices, W, H):
    pts = [_pt(lm, i, W, H) for i in indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h  = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def _estimate_emotion(lm, W, H) -> tuple:
    """Return (dominant_label, {label: score 0-100}) from 478-point mesh."""
    avg_ear     = (_ear(lm, _L_EYE, W, H) + _ear(lm, _R_EYE, W, H)) / 2

    cl  = _pt(lm, _MOUTH_CORNER_L, W, H)
    cr  = _pt(lm, _MOUTH_CORNER_R, W, H)
    mt  = _pt(lm, _MOUTH_TOP,      W, H)
    mb  = _pt(lm, _MOUTH_BOT,      W, H)
    nose = _pt(lm, _NOSE_TIP,      W, H)

    mouth_gap   = np.linalg.norm(mt - mb)
    lip_mid_y   = (mt[1] + mb[1]) / 2
    avg_corner_y = (cl[1] + cr[1]) / 2
    smile        = lip_mid_y - avg_corner_y     # + = corners above centre

    brow_y = np.mean([_pt(lm, i, W, H)[1] for i in _L_BROW + _R_BROW])
    eye_y  = np.mean([_pt(lm, i, W, H)[1] for i in [33, 133, 362, 263]])
    brow_offset = brow_y - eye_y               # smaller = brows closer to eyes

    # Normalised features [0, 1]
    eye_wide   = float(np.clip((avg_ear   - 0.20) / 0.18, 0, 1))
    mouth_open = float(np.clip((mouth_gap -  3.0) / 25.0, 0, 1))
    smiling    = float(np.clip(smile  /  8.0, 0, 1))
    frowning   = float(np.clip(-smile /  6.0, 0, 1))
    brows_low  = float(np.clip((20 - brow_offset) / 20.0, 0, 1))

    scores = {
        "HAPPY":     np.clip(smiling * 100, 0, 100),
        "SAD":       np.clip(frowning * 70 + (1 - eye_wide) * 30, 0, 100),
        "ANGRY":     np.clip(brows_low * 60 + frowning * 40, 0, 100),
        "SURPRISED": np.clip(eye_wide * 60 + mouth_open * 40, 0, 100),
        "FEAR":      np.clip(eye_wide * 50 + frowning * 30 + mouth_open * 20, 0, 100),
        "DISGUSTED": np.clip(brows_low * 40 + frowning * 60, 0, 100),
        "NEUTRAL":   np.clip((1 - smiling) * (1 - frowning) * (1 - brows_low) * 100, 0, 100),
    }
    scores["NEUTRAL"] = max(scores["NEUTRAL"], 10.0)
    scores = {k: float(v) for k, v in scores.items()}
    dominant = max(scores, key=lambda k: scores[k])
    return dominant, scores


# ── Draw helpers ──────────────────────────────────────────────────────────────

def _corners(img, x, y, w, h, color, arm=24, t=3):
    for (ax, ay), (bx, by), (cx, cy) in [
        ((x,   y),   (x+arm, y),   (x,   y+arm)),
        ((x+w, y),   (x+w-arm, y), (x+w, y+arm)),
        ((x,   y+h), (x+arm, y+h), (x,   y+h-arm)),
        ((x+w, y+h), (x+w-arm, y+h), (x+w, y+h-arm)),
    ]:
        cv2.line(img, (ax, ay), (bx, by), color, t, cv2.LINE_AA)
        cv2.line(img, (ax, ay), (cx, cy), color, t, cv2.LINE_AA)


def _pill(img, text, x, y, fg, scale=0.52, thick=1):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    p = 4
    cv2.rectangle(img, (x-p, y-th-p), (x+tw+p, y+bl+1), (12, 12, 12), -1)
    cv2.rectangle(img, (x-p, y-th-p), (x+tw+p, y+bl+1), fg, 1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def _bars(img, scores: dict, ox, oy, bw=108, bh=12):
    for i, (emo, pct) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1])):
        by  = oy + i * (bh + 6)
        col = EMOTION_COLORS.get(emo, DEFAULT_COLOR)
        cv2.rectangle(img, (ox, by), (ox+bw, by+bh), (38, 38, 38), -1)
        fill = max(1, int(bw * pct / 100))
        cv2.rectangle(img, (ox, by), (ox+fill, by+bh), col, -1)
        cv2.putText(img, f"{emo[:8]:<8} {pct:4.0f}%",
                    (ox+bw+6, by+bh-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)


# ── InsightFace age/gender wrapper ────────────────────────────────────────────

class _AgeGender:
    """Lazy-loaded InsightFace age+gender detector (runs in a background thread)."""

    def __init__(self):
        self._app   = None
        self._ready = False
        self._lock  = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        if not _IF_AVAILABLE:
            return
        try:
            print("[INFO] Loading InsightFace buffalo_s (~75 MB on first run)...")
            a = _IFApp(name="buffalo_s", providers=["CPUExecutionProvider"])
            a.prepare(ctx_id=0, det_size=(320, 320))
            with self._lock:
                self._app, self._ready = a, True
            print("[INFO] InsightFace ready.")
        except Exception as e:
            print(f"[WARN] InsightFace load failed: {e}")

    def analyse(self, bgr_img) -> list:
        """Returns [{bbox, age, gender}, …] or []."""
        with self._lock:
            if not self._ready:
                return []
            app = self._app
        try:
            faces = app.get(bgr_img)
            out = []
            for f in faces:
                out.append({
                    "bbox":   f.bbox.astype(int).tolist(),
                    "age":    int(getattr(f, "age", 0)),
                    "gender": getattr(f, "sex", "?") or "?",
                })
            return out
        except Exception:
            return []


# ── MediaPipe FaceLandmarker wrapper ──────────────────────────────────────────

class _Landmarker:
    """Wraps the MediaPipe Tasks FaceLandmarker (new API, mediapipe >= 0.10)."""

    def __init__(self):
        self._lmk   = None
        self._ready = False
        self._lock  = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        if not _MP_AVAILABLE:
            return
        if not _ensure_landmark_model():
            return
        try:
            BaseOptions = _mp_python.BaseOptions
            FLOptions   = _mp_vision.FaceLandmarkerOptions
            FL          = _mp_vision.FaceLandmarker

            options = FLOptions(
                base_options=BaseOptions(model_asset_path=_MODEL_PATH),
                num_faces=4,
                min_face_detection_confidence=0.45,
                min_face_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            lmk = FL.create_from_options(options)
            with self._lock:
                self._lmk, self._ready = lmk, True
            print("[INFO] MediaPipe FaceLandmarker ready.")
        except Exception as e:
            print(f"[WARN] FaceLandmarker load failed: {e}")

    def detect(self, rgb_img) -> list:
        """Returns list-of-lists: [[landmark, …], …] (one per face)."""
        with self._lock:
            if not self._ready:
                return []
            lmk = self._lmk
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            result = lmk.detect(mp_img)
            return [fl.landmark for fl in (result.face_landmarks or [])]
        except Exception:
            return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    age_gender  = _AgeGender()
    landmarker  = _Landmarker()

    # ── Camera init with permission guidance ──────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    # Wait up to 8 s for macOS to grant permission and the camera to warm up
    _deadline = time.time() + 8
    while time.time() < _deadline:
        ret, _ = cap.read()
        if ret:
            break
        print("[INFO] Waiting for camera... (grant access in"
              " System Settings > Privacy & Security > Camera if prompted)")
        time.sleep(1)
    else:
        print()
        print("  ERROR: Cannot open camera.")
        print("  On macOS you must grant camera access to Terminal / iTerm2:")
        print("  System Settings > Privacy & Security > Camera")
        print("  Then re-run the script.")
        cap.release()
        return

    fps_q   = deque(maxlen=30)
    prev_t  = time.time()
    frame_n = 0

    # Per-face caches
    ag_cache: dict = {}   # face_id -> {age, gender}
    ag_lock         = threading.Lock()
    ag_busy         = False

    def _ag_worker(img, ids):
        nonlocal ag_busy
        results = age_gender.analyse(img)
        with ag_lock:
            for i, r in enumerate(results):
                if i < len(ids):
                    ag_cache[ids[i]] = {"age": r["age"], "gender": r["gender"]}
        ag_busy = False

    print("=" * 62)
    print("  Face Tracker  |  Mood + Age + Gender")
    print("  Models will auto-download on first run.")
    print("  Press  Q  to quit.")
    print("=" * 62)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame dropped — retrying...")
            time.sleep(0.05)
            continue

        now = time.time()
        fps_q.append(1.0 / max(now - prev_t, 1e-9))
        prev_t  = now
        frame_n += 1
        fps     = sum(fps_q) / len(fps_q)

        H, W   = frame.shape[:2]
        display = frame.copy()
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Haar face detection (fast, always available) ──────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=5, minSize=(70, 70)
        )

        # ── Landmark detection ────────────────────────────────────────────────
        lm_faces = landmarker.detect(rgb)   # [[lm, …], …]

        # ── Age/gender background update every 25 frames ──────────────────────
        if frame_n % 25 == 0 and not ag_busy and len(boxes) > 0:
            ag_busy = True
            ids = [f"f{i}" for i in range(len(boxes))]
            threading.Thread(target=_ag_worker,
                             args=(frame.copy(), ids), daemon=True).start()

        # ── Render each face ──────────────────────────────────────────────────
        for i, (x, y, w, h) in enumerate(boxes):
            fid = f"f{i}"
            cx, cy = x + w // 2, y + h // 2

            # Match the closest landmark face to this box centre
            emotion_label  = "LOADING..."
            emotion_scores = {}
            best_lm, best_d = None, float("inf")
            for lm in lm_faces:
                nx = lm[_NOSE_TIP].x * W
                ny = lm[_NOSE_TIP].y * H
                d  = abs(nx - cx) + abs(ny - cy)
                if d < best_d:
                    best_d, best_lm = d, lm
            if best_lm and best_d < w * 1.5:
                emotion_label, emotion_scores = _estimate_emotion(best_lm, W, H)

            # Cached age/gender
            with ag_lock:
                ag = ag_cache.get(fid, {})
            age    = ag.get("age",    None)
            gender = ag.get("gender", None)

            color = EMOTION_COLORS.get(emotion_label, DEFAULT_COLOR)

            # Subtle tint inside box
            ov = display.copy()
            cv2.rectangle(ov, (x, y), (x+w, y+h), color, -1)
            cv2.addWeighted(ov, 0.07, display, 0.93, 0, display)

            # Border + corner brackets
            cv2.rectangle(display, (x, y), (x+w, y+h), (40, 40, 40), 1)
            _corners(display, x, y, w, h, color)

            # Face-centre crosshair
            cv2.drawMarker(display, (cx, cy), color,
                           cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

            # ── Pill labels ───────────────────────────────────────────────────
            lx = x
            ly = y - 14 if y > 90 else y + h + 22

            _pill(display, f"MOOD    {emotion_label}", lx, ly,    color)

            age_txt = f"AGE     ~{age} yrs" if age else "AGE     loading..."
            _pill(display, age_txt,                   lx, ly+26, (155, 210, 255))

            if gender == "M":
                g_col, g_txt = (80, 190, 255), "GENDER  Male"
            elif gender == "F":
                g_col, g_txt = (255, 150, 200), "GENDER  Female"
            else:
                g_col, g_txt = (170, 170, 170), "GENDER  loading..."
            _pill(display, g_txt,                     lx, ly+52, g_col)
            _pill(display, f"POS     ({cx}, {cy})",   lx, ly+78, (150, 150, 150))

            # Emotion confidence bars (right of face when room available)
            if emotion_scores:
                bx = x + w + 12
                if bx + 240 < W:
                    _bars(display, emotion_scores, bx, y)

        # ── HUD ──────────────────────────────────────────────────────────────
        cv2.rectangle(display, (0, 0), (W, 38), (8, 8, 8), -1)
        cv2.putText(
            display,
            f"FACE TRACKER  |  faces: {len(boxes)}"
            f"  |  fps: {fps:4.1f}"
            f"  |  frame: {frame_n}"
            f"  |  [Q] quit",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 175), 1, cv2.LINE_AA,
        )

        # Screen-centre guide cross
        cv2.line(display, (W//2-12, H//2), (W//2+12, H//2), (50, 50, 50), 1)
        cv2.line(display, (W//2, H//2-12), (W//2, H//2+12), (50, 50, 50), 1)

        cv2.imshow("Face Tracker  —  Mood & Age", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    # Explicitly close the landmarker to avoid mediapipe __del__ crash
    with landmarker._lock:
        if landmarker._lmk is not None:
            try:
                landmarker._lmk.close()
            except Exception:
                pass
            landmarker._lmk = None
    print("Tracker stopped.")


if __name__ == "__main__":
    main()
