"""
Face Tracker v4 — 14-Emotion Detection + Voice Recognition
============================================================
• Face detection    : OpenCV Haar Cascade
• Age + Gender      : InsightFace buffalo_s  (ONNX)
• Face recognition  : InsightFace normed_embedding  (512-dim MobileFaceNet)
• Emotion (14)      : MediaPipe 52 ARKit blendshapes → muscle-group mapping
    HAPPY · EXCITED · AMUSED · SAD · ANGRY · SURPRISED · SHOCKED
    FEAR · DISGUSTED · CONTEMPT · CONFUSED · CONCENTRATING · BORED · NEUTRAL
• Head pose         : Euler angles from MediaPipe transformation matrix
• Voice output      : macOS `say` command  (non-blocking TTS)
• Voice name input  : Prompted via terminal + spoken confirmation
• Persistence       : ~/.face_tracker/known_faces.pkl

Workflow
--------
  New face detected  → "Hello! I see a new face. Type your name + Enter."
  Name entered       → "Nice to meet you, <Name>! I'll remember you."
  Known face returns → "I can see you <Name>! You look <emotion> today!"

Press Q to quit.
"""

import cv2
import numpy as np
import threading
import time
import os
import urllib.request
import pickle
import queue
import random
import subprocess
import wave
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FONT   = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX

ACCENT   = (0,   210, 160)
WHITE    = (245, 245, 245)
DIM      = (115, 120, 130)
BLACK    = (0,   0,   0)
PANEL_BG = (12,  12,  18)
GREEN    = (0,   210,  90)
YELLOW   = (0,   210, 230)
RED_DIM  = (60,  60,  200)

# ── 14 emotions: colour (BGR) ─────────────────────────────────────────────────
EMOTION_COLORS = {
    # Positive
    "HAPPY":         (0,   215,  80),   # bright green
    "EXCITED":       (0,   230, 200),   # lime-yellow
    "AMUSED":        (60,  200,  80),   # soft green
    # Negative
    "SAD":           (200,  55,   0),   # deep blue-orange
    "ANGRY":         (0,    15, 230),   # vivid red
    "DISGUSTED":     (0,   110,  35),   # dark green
    "CONTEMPT":      (20,   70, 200),   # brick-orange
    "BORED":         (105, 108, 118),   # muted slate
    # Aroused / mixed
    "SURPRISED":     (0,   175, 255),   # amber
    "SHOCKED":       (0,   240, 250),   # bright yellow
    "FEAR":          (160,   0, 160),   # purple
    # Cognitive
    "CONFUSED":      (0,   160, 230),   # warm orange
    "CONCENTRATING": (175, 110,   0),   # steel blue
    # Baseline
    "NEUTRAL":       (170, 170, 170),   # neutral grey
}

# Ordered for consistent bar rendering (positive → negative → aroused → cognitive)
EMOTIONS = [
    "HAPPY", "EXCITED", "AMUSED",
    "SAD", "ANGRY", "DISGUSTED", "CONTEMPT", "BORED",
    "SURPRISED", "SHOCKED", "FEAR",
    "CONFUSED", "CONCENTRATING",
    "NEUTRAL",
]

RECOGNITION_THRESHOLD = 0.38
GREETING_COOLDOWN_S   = 25

# Anti-noise gating for the "ask for name" prompt:
ASK_MIN_FACE_PX        = 130   # face must be at least this wide AND tall
ASK_EDGE_MARGIN_PX     = 14    # face must not touch the frame edge
ASK_MAX_HEAD_ANGLE     = 25    # |yaw|, |pitch| must be ≤ this many degrees
ASK_STABLE_FRAMES      = 5     # consecutive UNKNOWN cycles before prompting
ASK_FAILURE_COOLDOWN_S = 15    # cooldown after a skip / unrecognised attempt
ASK_RECENT_SIM_THRESH  = 0.40  # don't re-prompt if embedding ~matches a recent ask
ASK_RECENT_TTL_S       = 60    # how long a recent-ask embedding stays remembered

# Spoken phrases per emotion (variety for naturalness)
_EMO_PHRASES = {
    "HAPPY":         ["really happy",        "full of joy",          "absolutely joyful",   "so cheerful"],
    "EXCITED":       ["very excited",         "full of excitement",   "super pumped",        "buzzing with energy"],
    "AMUSED":        ["amused",               "quite tickled",        "finding something funny", "entertained"],
    "SAD":           ["a bit sad",            "feeling down",         "a little blue",       "not so cheerful"],
    "ANGRY":         ["quite angry",          "a bit frustrated",     "upset about something","a little mad"],
    "DISGUSTED":     ["disgusted",            "put off by something", "displeased",          "a bit repulsed"],
    "CONTEMPT":      ["a bit contemptuous",   "somewhat scornful",    "a little smug",       "slightly dismissive"],
    "BORED":         ["a bit bored",          "looking tired",        "not very engaged",    "a little sleepy"],
    "SURPRISED":     ["quite surprised",      "caught off guard",     "rather startled",     "taken aback"],
    "SHOCKED":       ["absolutely shocked",   "completely stunned",   "blown away",          "in total disbelief"],
    "FEAR":          ["a bit scared",         "quite nervous",        "rather anxious",      "a little afraid"],
    "CONFUSED":      ["confused",             "a bit puzzled",        "not quite sure about something", "perplexed"],
    "CONCENTRATING": ["deep in thought",      "very focused",         "concentrating hard",  "in the zone"],
    "NEUTRAL":       ["calm",                 "relaxed",              "pretty neutral",      "chilled out"],
}

_GREET_KNOWN = [
    "I can see you {name}! You look {emotion} today!",
    "Hey {name}! You seem to be feeling {emotion}.",
    "Welcome back {name}! You look {emotion} right now.",
    "Oh hi {name}! You are looking {emotion} today!",
]

# Landmark indices for face-mesh dot overlay (MediaPipe 478-point)
_MESH_INDICES = list(dict.fromkeys([
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
    336, 296, 334, 293, 300, 383, 353, 276, 283, 282, 295, 285,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95,
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 326,
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
        print("[INFO] Landmark model ready.")
        return True
    except Exception:
        try:
            import requests as _r
            r = _r.get(_MODEL_URL, timeout=30)
            r.raise_for_status()
            with open(_MODEL_PATH, "wb") as f:
                f.write(r.content)
            print("[INFO] Landmark model ready.")
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

_SR_OK = False
try:
    import speech_recognition as _sr
    import sounddevice as _sd
    _SR_OK = True
except Exception as e:
    print(f"[WARN] SpeechRecognition / sounddevice unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Speaker  (macOS `say` — non-blocking)
# ─────────────────────────────────────────────────────────────────────────────
class Speaker:
    """Non-blocking text-to-speech using the macOS `say` command."""

    VOICE = "Samantha"   # change to any `say -v ?` voice you prefer
    RATE  = 185          # words per minute

    def __init__(self):
        self._busy = threading.Event()
        self._proc: subprocess.Popen | None = None

    def say(self, text: str, interrupt: bool = False):
        if self._busy.is_set():
            if not interrupt:
                return
            self.stop()
        self._busy.set()

        def _run():
            try:
                self._proc = subprocess.Popen(
                    ["say", "-v", self.VOICE, "-r", str(self.RATE), text],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self._proc.wait()
            except FileNotFoundError:
                pass   # `say` not available (non-macOS)
            finally:
                self._busy.clear()
                self._proc = None

        threading.Thread(target=_run, daemon=True).start()

    def stop(self):
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._busy.clear()

    def wait(self, timeout: float = 10.0):
        """Block until any currently-speaking TTS finishes (or timeout)."""
        deadline = time.time() + timeout
        while self._busy.is_set() and time.time() < deadline:
            time.sleep(0.05)

    @property
    def is_busy(self) -> bool:
        return self._busy.is_set()


# ─────────────────────────────────────────────────────────────────────────────
# Voice input  (microphone → text via SpeechRecognition + Google Web API)
# ─────────────────────────────────────────────────────────────────────────────
_AUDIO_DIR = os.path.join(_MODEL_DIR, "audio")


class VoiceInput:
    """
    Records audio from the default microphone, saves it as WAV, and
    converts it to text using Google's free Web Speech API.

    Shared state (`status`) is read by the main render loop to draw a
    big "● RECORDING …" badge over the video while capture is active.
    """

    SAMPLE_RATE = 16000   # 16 kHz mono — what Google's API expects
    DURATION    = 4.0     # seconds per name capture
    LANGUAGE    = "en-US"

    def __init__(self):
        os.makedirs(_AUDIO_DIR, exist_ok=True)
        self._recog = _sr.Recognizer() if _SR_OK else None
        # Live state for the UI
        self.status: dict = {"phase": "idle",   # idle | listening | thinking | done | error
                             "ends_at": 0.0,
                             "text":   "",
                             "error":  ""}

    @property
    def available(self) -> bool:
        return _SR_OK and self._recog is not None

    def capture_name(self) -> tuple:
        """
        Blocking call (run in its own thread):
          1. record DURATION seconds of mono 16 kHz audio
          2. save WAV to ~/.face_tracker/audio/name_<ts>.wav
          3. transcribe with Google Web Speech API
          4. update self.status throughout for the UI
        Returns (recognized_text, wav_path) — text is "" on failure.
        """
        if not self.available:
            self.status.update(phase="error", error="SR unavailable")
            return "", ""

        # ── 1. Record ────────────────────────────────────────────────────────
        self.status.update(phase="listening",
                           ends_at=time.time() + self.DURATION,
                           text="", error="")
        try:
            audio = _sd.rec(int(self.DURATION * self.SAMPLE_RATE),
                            samplerate=self.SAMPLE_RATE,
                            channels=1, dtype="int16")
            _sd.wait()
        except Exception as e:
            self.status.update(phase="error", error=f"mic error: {e}")
            return "", ""

        # ── 2. Save WAV ──────────────────────────────────────────────────────
        ts        = int(time.time())
        wav_path  = os.path.join(_AUDIO_DIR, f"name_{ts}.wav")
        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)              # 16-bit PCM
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
        except Exception as e:
            print(f"[WARN] Could not save audio: {e}")
            wav_path = ""

        # ── 3. Transcribe ────────────────────────────────────────────────────
        self.status.update(phase="thinking")
        try:
            sr_audio = _sr.AudioData(audio.tobytes(), self.SAMPLE_RATE, 2)
            text     = self._recog.recognize_google(sr_audio,
                                                    language=self.LANGUAGE)
            text     = text.strip()
            self.status.update(phase="done", text=text, ends_at=time.time())
            # Persist transcript next to the WAV for traceability
            if wav_path:
                try:
                    with open(wav_path.replace(".wav", ".txt"), "w") as f:
                        f.write(text + "\n")
                except Exception:
                    pass
            print(f"[VOICE] Heard: '{text}'  → {wav_path}")
            return text, wav_path
        except _sr.UnknownValueError:
            self.status.update(phase="error", error="couldn't understand",
                               ends_at=time.time())
            print("[VOICE] Could not understand the audio.")
            return "", wav_path
        except _sr.RequestError as e:
            self.status.update(phase="error", error="no internet",
                               ends_at=time.time())
            print(f"[VOICE] STT service error (offline?): {e}")
            return "", wav_path
        except Exception as e:
            self.status.update(phase="error", error=str(e)[:30],
                               ends_at=time.time())
            return "", wav_path


# ─────────────────────────────────────────────────────────────────────────────
# Known-faces store  (persistent)
# ─────────────────────────────────────────────────────────────────────────────
_FACES_FILE = os.path.join(_MODEL_DIR, "known_faces.pkl")


class KnownFaces:
    """Stores {name, normed_embedding} pairs and does cosine similarity lookup."""

    def __init__(self):
        os.makedirs(_MODEL_DIR, exist_ok=True)
        self._faces: list[dict] = self._load()
        self._lock = threading.Lock()

    def _load(self) -> list:
        if os.path.exists(_FACES_FILE):
            try:
                with open(_FACES_FILE, "rb") as f:
                    data = pickle.load(f)
                print(f"[INFO] Loaded {len(data)} known face(s).")
                return data
            except Exception:
                pass
        return []

    def save(self):
        with open(_FACES_FILE, "wb") as f:
            pickle.dump(self._faces, f)

    def find(self, emb: np.ndarray) -> tuple:
        """Returns (name, similarity) or (None, 0.0)."""
        with self._lock:
            if emb is None or len(self._faces) == 0:
                return None, 0.0
            best_name, best_sim = None, 0.0
            for rec in self._faces:
                sim = float(np.dot(emb, rec["embedding"]))   # both L2-normed
                if sim > best_sim:
                    best_sim, best_name = sim, rec["name"]
            if best_sim >= RECOGNITION_THRESHOLD:
                return best_name, best_sim
            return None, best_sim

    def add(self, name: str, emb: np.ndarray):
        with self._lock:
            self._faces.append({"name": name, "embedding": emb})
            self.save()
        print(f"[INFO] Saved face for '{name}' ({len(self._faces)} total).")

    def all_names(self) -> list:
        with self._lock:
            return [r["name"] for r in self._faces]

    def clear(self) -> int:
        """Delete every known face. Returns the count that was removed."""
        with self._lock:
            n = len(self._faces)
            self._faces.clear()
            try:
                if os.path.exists(_FACES_FILE):
                    os.remove(_FACES_FILE)
            except Exception:
                pass
        print(f"[INFO] Cleared {n} known face(s).")
        return n


# ─────────────────────────────────────────────────────────────────────────────
# Blendshape → emotion  (52 ARKit coefficients)
# ─────────────────────────────────────────────────────────────────────────────
def _bs_dict(blendshapes) -> dict:
    out = {}
    if blendshapes is None:
        return out
    for item in blendshapes:
        name  = getattr(item, "category_name", None) or getattr(item, "label", "")
        score = getattr(item, "score", 0.0)
        out[name] = float(score)
    return out


def blendshapes_to_scores(blendshapes) -> dict:
    """
    Map 52 ARKit blendshape coefficients to 14 emotion scores [0-100].

    Each emotion is grounded in distinct facial action units (AUs):
      HAPPY         – bilateral zygomatic smile + Duchenne cheek squint
      EXCITED       – big smile + wide eyes + open mouth  (high arousal)
      AMUSED        – modest smile + knowing orbital squint (low arousal)
      SAD           – depressor frown + corrugator brow raise + lip pull-down
      ANGRY         – corrugator brow pull-down + orbicularis squint + frown
      DISGUSTED     – levator nasi sneer + frown + brow compression
      CONTEMPT      – *asymmetric* unilateral sneer / half-smile
      BORED         – partial lid droop + slack, low-activation face
      SURPRISED     – frontalis brow raise + eye-wide + jaw drop
      SHOCKED       – all three surprise AUs simultaneously at peak
      FEAR          – medial brow raise + scleral show + mouth stretch
      CONFUSED      – *asymmetric* brow (one up / one down) + mild squint
      CONCENTRATING – *symmetric* brow compression + squint + pressed lips
      NEUTRAL       – residual when all others are low
    """
    bs = _bs_dict(blendshapes)

    def g(*keys):
        return sum(bs.get(k, 0) for k in keys) / max(len(keys), 1)

    def asym(lk, rk):
        return abs(bs.get(lk, 0) - bs.get(rk, 0))

    # ── Primary AU groups ────────────────────────────────────────────────────
    smile_l      = bs.get("mouthSmileLeft",  0)
    smile_r      = bs.get("mouthSmileRight", 0)
    smile_avg    = (smile_l + smile_r) / 2
    smile_asym   = abs(smile_l - smile_r)

    frown_avg    = g("mouthFrownLeft",    "mouthFrownRight")
    cheek_sq     = g("cheekSquintLeft",   "cheekSquintRight")
    brow_down    = g("browDownLeft",      "browDownRight")
    brow_dn_asym = asym("browDownLeft",   "browDownRight")
    brow_inner   = bs.get("browInnerUp", 0)
    brow_outer   = g("browOuterUpLeft",   "browOuterUpRight")
    brow_ou_asym = asym("browOuterUpLeft","browOuterUpRight")
    eye_wide     = g("eyeWideLeft",       "eyeWideRight")
    eye_squint   = g("eyeSquintLeft",     "eyeSquintRight")
    eye_blink    = g("eyeBlinkLeft",      "eyeBlinkRight")
    jaw_open     = bs.get("jawOpen", 0)
    nose_sneer   = g("noseSneerLeft",     "noseSneerRight")
    nose_asym    = asym("noseSneerLeft",  "noseSneerRight")
    mouth_press  = g("mouthPressLeft",    "mouthPressRight")
    mouth_str    = g("mouthStretchLeft",  "mouthStretchRight")
    lip_lower    = g("mouthLowerDownLeft","mouthLowerDownRight")
    brow_asym    = brow_dn_asym + brow_ou_asym

    # ── 14 raw scores ────────────────────────────────────────────────────────
    happy    = smile_avg * 0.60 + cheek_sq * 0.40

    excited  = smile_avg * 0.42 + eye_wide * 0.30 + jaw_open * 0.18 \
             + brow_outer * 0.10

    # AMUSED: less smile than HAPPY, more eye squint (knowing look)
    amused   = min(smile_avg, 0.55) * 0.46 + eye_squint * 0.34 + cheek_sq * 0.20

    sad      = frown_avg * 0.48 + brow_inner * 0.32 + lip_lower * 0.20

    angry    = brow_down * 0.52 + eye_squint * 0.28 + frown_avg * 0.20

    disgusted = nose_sneer * 0.52 + frown_avg * 0.30 + brow_down * 0.18

    # CONTEMPT: unilateral marker — asymmetry is the key discriminator
    contempt = nose_asym * 0.48 + smile_asym * 0.38 \
             + max(nose_sneer - 0.08, 0) * 0.14

    # BORED: partial lid droop — scale so half-closed eyes (blink≈0.5) peak at 1.0
    eye_droop = min(eye_blink * (1.0 - eye_blink) * 4.5, 1.0)
    bored    = eye_droop * 0.55 + max(0.30 - smile_avg, 0) * 0.25 \
             + max(0.30 - brow_inner, 0) * 0.20

    surprised = jaw_open * 0.40 + eye_wide * 0.36 + brow_outer * 0.24

    # SHOCKED: requires ALL THREE surprise AUs to exceed a high floor simultaneously.
    # Using (x-floor)*(y-floor)*(z-floor) so it stays near zero for "merely surprised"
    # and spikes only when jaw + eyes + brows are all at extreme amplitude.
    _sj = max(jaw_open   - 0.55, 0)
    _se = max(eye_wide   - 0.55, 0)
    _sb = max(brow_outer - 0.48, 0)
    shocked  = min(_sj * _se * _sb * 120, 1.0)

    fear     = brow_inner * 0.32 + eye_wide * 0.30 + mouth_str * 0.22 \
             + jaw_open * 0.16

    # CONFUSED: asymmetric brow activity is the main signal
    confused = min(brow_asym * 1.55, 1.0) * 0.52 + eye_squint * 0.28 \
             + min(frown_avg, 0.35) * 0.20

    # CONCENTRATING: symmetric brow pull-down; subtract asymmetry so it
    # doesn't overlap with CONFUSED
    concentrating = max(brow_down - brow_dn_asym * 0.65, 0) * 0.44 \
                  + eye_squint * 0.34 + mouth_press * 0.22

    raw_peak = max(happy, excited, amused, sad, angry, disgusted, contempt,
                   bored, surprised, shocked, fear, confused, concentrating)
    neutral  = max(0.0, 1.0 - raw_peak * 1.80)

    raw = {
        "HAPPY": happy, "EXCITED": excited, "AMUSED": amused,
        "SAD": sad, "ANGRY": angry, "DISGUSTED": disgusted,
        "CONTEMPT": contempt, "BORED": bored,
        "SURPRISED": surprised, "SHOCKED": shocked, "FEAR": fear,
        "CONFUSED": confused, "CONCENTRATING": concentrating,
        "NEUTRAL": neutral,
    }
    raw  = {k: max(v, 0.0) for k, v in raw.items()}
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
    ratio = face_w / max(frame_w, 1)
    return int(0.25 * 65 / max(ratio, 0.01))


# ─────────────────────────────────────────────────────────────────────────────
# Smoothed face state
# ─────────────────────────────────────────────────────────────────────────────
class FaceState:
    def __init__(self):
        self.scores   = {e: 100 / len(EMOTIONS) for e in EMOTIONS}
        self.timeline = deque(maxlen=180)

    def smooth(self, new_scores: dict, alpha=0.28):
        for k in EMOTIONS:
            self.scores[k] = alpha * new_scores.get(k, 0) \
                           + (1 - alpha) * self.scores.get(k, 0)
        self.timeline.append(self.dominant)

    @property
    def dominant(self) -> str:
        return max(self.scores, key=self.scores.__getitem__)


# ─────────────────────────────────────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────────────────────────────────────
def _txt(img, text, x, y, color=WHITE, scale=0.58, thick=1):
    cv2.putText(img, text, (x+1, y+1), FONT, scale, BLACK, thick+2, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), FONT, scale, color, thick,   cv2.LINE_AA)


def _stxt(img, text, x, y, color=DIM, scale=0.42, thick=1):
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


def _corners(img, x, y, w, h, color, arm=30, thick=3, anim=0.0):
    a = int(arm + anim * 7)
    segs = [
        ((x,   y),   (x+a, y)),    ((x,   y),   (x,   y+a)),
        ((x+w, y),   (x+w-a, y)),  ((x+w, y),   (x+w, y+a)),
        ((x,   y+h), (x+a, y+h)),  ((x,   y+h), (x,   y+h-a)),
        ((x+w, y+h), (x+w-a, y+h)),((x+w, y+h), (x+w, y+h-a)),
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


def _emotion_bars(img, scores, ox, oy, bw=108, bh=10, gap=4):
    """Draw compact horizontal bars for all 14 emotions, sorted by score."""
    sorted_emos = sorted(EMOTIONS, key=lambda e: scores.get(e, 0), reverse=True)
    for i, emo in enumerate(sorted_emos):
        by  = oy + i * (bh + gap)
        col = EMOTION_COLORS.get(emo, DIM)
        pct = scores.get(emo, 0)
        _hbar(img, ox, by, bw, bh, pct, col)
        _stxt(img, f"{emo[:8]:<8}  {pct:4.0f}%",
              ox + bw + 5, by + bh - 1, col, scale=0.36)


def _pose_compass(img, yaw, pitch, cx, cy, r=28):
    cv2.circle(img, (cx, cy), r, (45, 50, 60), 1, cv2.LINE_AA)
    cv2.line(img, (cx-r, cy), (cx+r, cy), (35, 40, 50), 1)
    cv2.line(img, (cx, cy-r), (cx, cy+r), (35, 40, 50), 1)
    dx = int(np.clip(np.sin(np.radians(yaw)),   -1, 1) * r * 0.9)
    dy = int(np.clip(np.sin(np.radians(pitch)), -1, 1) * r * 0.9)
    cv2.circle(img, (cx + dx, cy + dy), 5, (180, 220, 255), -1, cv2.LINE_AA)
    cv2.line(img, (cx, cy), (cx+dx, cy+dy), (120, 160, 200), 1, cv2.LINE_AA)


def _timeline(img, history, x, y, W, bar_h=14):
    tl = list(history)
    _panel(img, x, y, W, bar_h + 22, alpha=0.80, border=(45, 50, 62))
    _stxt(img, "EMOTION  TIMELINE", x+6, y+13, DIM, scale=0.38)
    if not tl:
        return
    bw = max(1, W // max(len(tl), 1))
    for i, emo in enumerate(tl):
        bx = x + i * bw
        cv2.rectangle(img, (bx, y+18), (bx+bw, y+18+bar_h),
                      EMOTION_COLORS.get(emo, DIM), -1)


def _speech_bubble(img, text, x, y, color, pulse):
    """Draw an animated speech-bubble label."""
    alpha = 0.55 + pulse * 0.30
    _stxt(img, text, x, y, color, scale=0.50)


# ─────────────────────────────────────────────────────────────────────────────
# InsightFace — age, gender, AND face embedding
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

    def analyse(self, bgr_img) -> list:
        """Returns [{bbox, age, gender, embedding}, …] sorted by x-centre."""
        with self._lock:
            if not self._ready:
                return []
            app = self._app
        try:
            faces = app.get(bgr_img)
            out = []
            for f in faces:
                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    emb = getattr(f, "embedding", None)
                out.append({
                    "bbox":      f.bbox.astype(int).tolist(),
                    "age":       int(getattr(f, "age", 0)),
                    "gender":    getattr(f, "sex", "?") or "?",
                    "embedding": np.array(emb) if emb is not None else None,
                })
            # sort left-to-right so face IDs stay consistent with Haar boxes
            out.sort(key=lambda r: (r["bbox"][0] + r["bbox"][2]) / 2)
            return out
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe FaceLandmarker
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
            with self._lock:
                self._lmk = _mpv.FaceLandmarker.create_from_options(opts)
                self._ready = True
            print("[INFO] MediaPipe FaceLandmarker ready.")
        except Exception as e:
            print(f"[WARN] FaceLandmarker: {e}")

    def detect(self, rgb_img) -> list:
        with self._lock:
            if not self._ready:
                return []
            lmk = self._lmk
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            r      = lmk.detect(mp_img)
            out    = []
            for i, fl in enumerate(r.face_landmarks or []):
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
# Greeting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _emotion_phrase(emotion: str) -> str:
    opts = _EMO_PHRASES.get(emotion, ["interesting"])
    return random.choice(opts)


def _greet_known(name: str, emotion: str) -> str:
    phrase = _emotion_phrase(emotion)
    tmpl   = random.choice(_GREET_KNOWN)
    return tmpl.format(name=name, emotion=phrase)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cascade    = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    ag_det     = AgeGender()
    landmarker = Landmarker()
    speaker    = Speaker()
    known      = KnownFaces()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    deadline = time.time() + 8
    while time.time() < deadline:
        ok, _ = cap.read()
        if ok:
            break
        print("[INFO] Waiting for camera… "
              "(System Settings > Privacy & Security > Camera)")
        time.sleep(1)
    else:
        print("\nERROR: Cannot open camera.")
        cap.release()
        return

    fps_q   = deque(maxlen=30)
    prev_t  = time.time()
    frame_n = 0

    # ── Age/gender/embedding background cache ─────────────────────────────────
    ag_cache: dict = {}    # fid -> {age, gender, embedding}
    ag_lock  = threading.Lock()
    ag_busy  = False

    def _ag_worker(img, haar_boxes):
        nonlocal ag_busy
        results = ag_det.analyse(img)
        with ag_lock:
            for j, r in enumerate(results):
                # Match IF face to the nearest Haar box by x-centre
                if_cx = (r["bbox"][0] + r["bbox"][2]) / 2
                best_i, best_d = 0, float("inf")
                for bi, (hx, hy, hw, hh) in enumerate(haar_boxes):
                    d = abs(if_cx - (hx + hw / 2))
                    if d < best_d:
                        best_d, best_i = d, bi
                ag_cache[f"f{best_i}"] = r
        ag_busy = False

    # ── Per-face smooth state ─────────────────────────────────────────────────
    face_states: dict[str, FaceState] = {}

    # ── Recognition state ─────────────────────────────────────────────────────
    # fid -> {"name": str|None, "sim": float, "status": "KNOWN"|"UNKNOWN"|"ASKING"}
    recog: dict = {}
    last_greeted: dict = {}   # name -> timestamp

    # ── Name-input pipeline (voice) ───────────────────────────────────────────
    voice_in  = VoiceInput()
    name_q    = queue.Queue()   # background thread drops name here
    asking_fid: str | None = None

    # Anti-noise gating state for the "ask" flow
    unknown_streak: dict = {}              # fid -> consecutive UNKNOWN cycles
    last_ask_failed_at: float = 0.0        # global cooldown after a skip
    recent_asks: list = []                 # [(timestamp, embedding), …]

    # Reset-confirmation state (two-press R)
    reset_pending_until: float = 0.0

    # Last per-fid head pose (set later in the render pass; we need it here
    # for the quality gate, so seed with zeros).
    head_pose: dict = {}                   # fid -> (yaw, pitch, roll)

    def _recently_asked(emb) -> bool:
        if emb is None:
            return False
        cutoff = time.time() - ASK_RECENT_TTL_S
        recent_asks[:] = [(t, e) for t, e in recent_asks if t > cutoff]
        for _, e in recent_asks:
            if float(np.dot(emb, e)) > ASK_RECENT_SIM_THRESH:
                return True
        return False

    def _good_quality_for_ask(box, frame_w, frame_h, yaw, pitch) -> bool:
        x, y, w, h = box
        if w < ASK_MIN_FACE_PX or h < ASK_MIN_FACE_PX:
            return False
        if x < ASK_EDGE_MARGIN_PX or y < ASK_EDGE_MARGIN_PX:
            return False
        if x + w > frame_w - ASK_EDGE_MARGIN_PX:
            return False
        if y + h > frame_h - ASK_EDGE_MARGIN_PX:
            return False
        if abs(yaw) > ASK_MAX_HEAD_ANGLE or abs(pitch) > ASK_MAX_HEAD_ANGLE:
            return False
        return True

    def _ask_name_thread():
        """Speak prompt → record audio → STT → enqueue name (or fall back to typing)."""
        nonlocal last_ask_failed_at
        speaker.say("Hello! Please say your name after the beep, "
                    "and I will remember you.", interrupt=True)
        speaker.wait()
        # Short audible beep so the user knows recording started
        subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.35)

        text, wav = voice_in.capture_name()
        if text:
            name_q.put(text)
            return

        # Voice failed → spoken apology + terminal fallback
        speaker.say("Sorry, I didn't catch that. "
                    "Please type your name in the terminal instead.")
        print("\n" + "═"*50)
        print("  Voice recognition failed.  TYPE the name + Enter:")
        print("  (or just press Enter to skip)")
        print("═"*50 + "\n>>> ", end="", flush=True)
        try:
            entered = input().strip()
        except EOFError:
            entered = ""
        if not entered:
            # Skipped → arm the failure cooldown so we don't immediately re-ask
            last_ask_failed_at = time.time()
        name_q.put(entered)

    print("=" * 65)
    print("  Face Tracker v4  |  Voice Capture + Recognition + 14 Emotions")
    print("  Q = quit       R (twice) = wipe ALL known faces")
    print(f"  Voice input : {'enabled' if voice_in.available else 'DISABLED (typing only)'}")
    print(f"  Audio dir   : {_AUDIO_DIR}")
    if known.all_names():
        print(f"  Known people: {', '.join(known.all_names())}")
    print("=" * 65)
    speaker.say("Face tracker started. Show me your face!", interrupt=True)

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
        anim    = (np.sin(frame_n * 0.10) + 1) / 2

        H, W   = frame.shape[:2]
        display = frame.copy()
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Detection ────────────────────────────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=5, minSize=(70, 70))
        lm_faces = landmarker.detect(rgb)

        # Age/gender/embedding every 25 frames
        if frame_n % 25 == 0 and not ag_busy and len(boxes) > 0:
            ag_busy = True
            threading.Thread(target=_ag_worker,
                             args=(frame.copy(), list(boxes)), daemon=True).start()

        # Purge dead face states & recognition
        active = {f"f{i}" for i in range(len(boxes))}
        for k in list(face_states):
            if k not in active:
                del face_states[k]
        for k in list(recog):
            if k not in active:
                del recog[k]
        for k in list(unknown_streak):
            if k not in active:
                del unknown_streak[k]
        for k in list(head_pose):
            if k not in active:
                del head_pose[k]
        if asking_fid and asking_fid not in active:
            asking_fid = None   # face went away while asking

        # ── Process name input from terminal ──────────────────────────────────
        if not name_q.empty():
            entered_name = name_q.get()
            if entered_name and asking_fid:
                with ag_lock:
                    ag = ag_cache.get(asking_fid, {})
                emb = ag.get("embedding")
                if emb is not None:
                    known.add(entered_name, emb)
                    recog[asking_fid] = {"name": entered_name,
                                         "sim": 1.0, "status": "KNOWN"}
                    speaker.say(f"Nice to meet you, {entered_name}! "
                                f"I'll remember you.", interrupt=True)
                    last_greeted[entered_name] = now
                else:
                    speaker.say(f"I couldn't capture your face properly. "
                                f"Try again in a moment.", interrupt=True)
            elif asking_fid:
                speaker.say("Okay, I'll skip that for now.", interrupt=True)
            asking_fid = None

        # ── Per-face: recognition + greeting ─────────────────────────────────
        for i, (x, y, w, h) in enumerate(boxes):
            fid    = f"f{i}"
            cx, cy = x + w // 2, y + h // 2

            if fid not in face_states:
                face_states[fid] = FaceState()

            with ag_lock:
                ag = ag_cache.get(fid, {})
            emb = ag.get("embedding")

            # Try recognition when we have an embedding and haven't yet classified
            if emb is not None and fid not in recog:
                name, sim = known.find(emb)
                if name:
                    recog[fid] = {"name": name, "sim": sim, "status": "KNOWN"}
                    # Greet (with cooldown per person name)
                    if now - last_greeted.get(name, 0) > GREETING_COOLDOWN_S \
                       and not speaker.is_busy:
                        dominant = face_states[fid].dominant
                        msg = _greet_known(name, dominant)
                        speaker.say(msg)
                        last_greeted[name] = now
                        print(f"[GREET] {msg}")
                else:
                    recog[fid] = {"name": None, "sim": sim, "status": "UNKNOWN"}

            # ── Anti-noise gating for the "ask" prompt ────────────────────────
            #   We only prompt if ALL of the following hold:
            #     1. status is UNKNOWN (face seen, embedding doesn't match anyone)
            #     2. no ask is currently in progress
            #     3. nothing queued from a previous ask
            #     4. embedding is available (else we couldn't save it anyway)
            #     5. face is good quality (big enough, not edge-clipped, frontal)
            #     6. UNKNOWN for ≥ ASK_STABLE_FRAMES consecutive cycles
            #     7. global cooldown after a previous skip has elapsed
            #     8. embedding wasn't already asked recently (no re-prompt loops)
            rec_status = recog.get(fid, {}).get("status", "LOADING")

            if rec_status == "UNKNOWN":
                unknown_streak[fid] = unknown_streak.get(fid, 0) + 1
            else:
                unknown_streak[fid] = 0

            yaw_f, pitch_f, _ = head_pose.get(fid, (0.0, 0.0, 0.0))

            if (rec_status == "UNKNOWN"
                    and asking_fid is None
                    and not name_q.qsize()
                    and emb is not None
                    and unknown_streak.get(fid, 0) >= ASK_STABLE_FRAMES
                    and (now - last_ask_failed_at) > ASK_FAILURE_COOLDOWN_S
                    and _good_quality_for_ask((x, y, w, h), W, H, yaw_f, pitch_f)
                    and not _recently_asked(emb)):
                asking_fid = fid
                recog[fid]["status"] = "ASKING"
                recent_asks.append((time.time(), emb.copy()))
                threading.Thread(target=_ask_name_thread, daemon=True).start()

            # If KNOWN face emotion changes significantly → spoken emotion update
            rec = recog.get(fid, {})
            if rec.get("status") == "KNOWN":
                name     = rec["name"]
                dominant = face_states[fid].dominant
                # Only re-greet if cooldown expired and not busy
                if now - last_greeted.get(name, 0) > GREETING_COOLDOWN_S \
                   and not speaker.is_busy and emb is not None:
                    msg = _greet_known(name, dominant)
                    speaker.say(msg)
                    last_greeted[name] = now
                    print(f"[GREET] {msg}")

        # ── Per-face render ───────────────────────────────────────────────────
        for i, (x, y, w, h) in enumerate(boxes):
            fid    = f"f{i}"
            cx, cy = x + w // 2, y + h // 2
            state  = face_states.get(fid, FaceState())
            rec    = recog.get(fid, {})

            # MediaPipe landmark match
            best, best_d = None, float("inf")
            for lfd in lm_faces:
                lms = lfd["landmarks"]
                nx, ny = lms[4].x * W, lms[4].y * H
                d = abs(nx - cx) + abs(ny - cy)
                if d < best_d:
                    best_d, best = d, lfd

            blendshapes = None
            matrix      = None
            landmarks   = None
            eye_l, eye_r, sm_pc = 80.0, 80.0, 0.0
            yaw = pitch = roll = 0.0

            if best and best_d < w * 1.5:
                landmarks   = best["landmarks"]
                blendshapes = best["blendshapes"]
                matrix      = best["matrix"]
                if blendshapes:
                    state.smooth(blendshapes_to_scores(blendshapes))
                    eye_l, eye_r = eye_open_pct(blendshapes)
                    sm_pc        = smile_pct(blendshapes)
                if matrix is not None:
                    yaw, pitch, roll = matrix_to_euler(matrix)

            # Cache head pose so the next frame's ask-gate can read it
            head_pose[fid] = (yaw, pitch, roll)

            scores   = dict(state.scores)
            dominant = state.dominant
            em_color = EMOTION_COLORS.get(dominant, DIM)

            # Box border color by recognition status
            status    = rec.get("status", "LOADING")
            box_color = {"KNOWN":   GREEN,
                         "ASKING":  YELLOW,
                         "UNKNOWN": RED_DIM,
                         "LOADING": ACCENT}.get(status, ACCENT)

            with ag_lock:
                ag = ag_cache.get(fid, {})
            age    = ag.get("age",    None)
            gender = ag.get("gender", None)
            dist   = est_distance(w, W)

            # ── Visuals ───────────────────────────────────────────────────────
            if landmarks:
                _face_mesh(display, landmarks, W, H, alpha=0.50)

            _scan_line(display, x, y, w, h, frame_n, box_color)

            ov = display.copy()
            cv2.rectangle(ov, (x, y), (x+w, y+h), box_color, -1)
            cv2.addWeighted(ov, 0.07, display, 0.93, 0, display)

            cv2.rectangle(display, (x, y), (x+w, y+h), (38, 42, 52), 1)
            _corners(display, x, y, w, h, box_color, arm=30, thick=3, anim=anim)
            cv2.drawMarker(display, (cx, cy), box_color,
                           cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)

            compass_cy = max(40, y - 44)
            _pose_compass(display, yaw, pitch, cx, compass_cy, r=28)

            # ── Info panel ────────────────────────────────────────────────────
            pw, ph = 218, 286
            px = x - pw - 10 if x - pw - 10 > 0 else x + w + 10
            py = max(40, min(y, H - ph - 6))
            _panel(display, px, py, pw, ph, border=box_color)

            lp  = px + 10
            row = py + 20

            # Name / status header
            rec_name = rec.get("name")
            if rec_name:
                _txt(display, rec_name.upper(), lp, row, GREEN, scale=0.68)
                sim = rec.get("sim", 0)
                _stxt(display, f"  match {sim*100:.0f}%",
                      lp + len(rec_name) * 14, row, DIM, scale=0.38)
            elif status == "ASKING":
                col = YELLOW if (frame_n // 15) % 2 == 0 else DIM
                _txt(display, "TELL ME YOUR NAME", lp, row, col, scale=0.50)
                _stxt(display, "type in terminal + Enter",
                      lp, row + 16, DIM, scale=0.36)
                row += 14
            elif status == "LOADING":
                _txt(display, f"FACE  #{i+1}", lp, row, ACCENT, scale=0.60)
            else:
                _txt(display, f"FACE  #{i+1}  UNKNOWN", lp, row, RED_DIM, scale=0.52)

            row += 8
            _divider(display, px, row, pw, box_color)
            row += 14

            # Mood
            _stxt(display, "MOOD", lp, row, DIM, scale=0.44)
            _txt( display, dominant, lp + 60, row, em_color, scale=0.56)
            row += 20
            conf = scores.get(dominant, 0)
            _hbar(display, lp, row, pw - 20, 7, conf, em_color)
            _stxt(display, f"{conf:.0f}%", lp + pw - 16, row + 6,
                  em_color, scale=0.38)
            row += 16

            # Smile
            _stxt(display, "SMILE", lp, row, DIM, scale=0.40)
            _hbar(display, lp + 56, row - 6, 76, 7, sm_pc,
                  EMOTION_COLORS["HAPPY"])
            _stxt(display, f"{sm_pc:.0f}%", lp + 138, row,
                  EMOTION_COLORS["HAPPY"], scale=0.38)
            row += 14
            _divider(display, px, row, pw)
            row += 12

            # Age + gender
            _stxt(display, "AGE",    lp,      row, DIM, scale=0.44)
            _txt( display, f"~{age} yrs" if age else "loading…",
                  lp + 60, row, WHITE, scale=0.52)
            row += 22
            _stxt(display, "GENDER", lp,      row, DIM, scale=0.44)
            if gender == "M":
                gc, gv = (80, 190, 255), "Male"
            elif gender == "F":
                gc, gv = (255, 150, 210), "Female"
            else:
                gc, gv = DIM, "loading…"
            _txt(display, gv, lp + 60, row, gc, scale=0.52)
            row += 22
            _divider(display, px, row, pw)
            row += 12

            # Head pose + distance
            _stxt(display, "YAW",   lp,       row, DIM,   scale=0.42)
            _stxt(display, f"{yaw:+.0f}°",  lp+40,  row, WHITE, scale=0.46)
            _stxt(display, "PITCH", lp+90,   row, DIM,   scale=0.42)
            _stxt(display, f"{pitch:+.0f}°",lp+142, row, WHITE, scale=0.46)
            row += 18
            _stxt(display, "ROLL",  lp,       row, DIM,   scale=0.42)
            _stxt(display, f"{roll:+.0f}°",  lp+40,  row, WHITE, scale=0.46)
            _stxt(display, "DIST",  lp+90,   row, DIM,   scale=0.42)
            _stxt(display, f"~{dist}cm",     lp+142, row, WHITE, scale=0.46)
            row += 18
            _divider(display, px, row, pw)
            row += 12

            # Eye bars
            _stxt(display, "L.EYE", lp, row, DIM, scale=0.40)
            _hbar(display, lp+56, row-6, 72, 7, eye_l, (0, 200, 220))
            _stxt(display, f"{eye_l:.0f}%", lp+134, row, (0, 200, 220), scale=0.38)
            row += 16
            _stxt(display, "R.EYE", lp, row, DIM, scale=0.40)
            _hbar(display, lp+56, row-6, 72, 7, eye_r, (0, 200, 220))
            _stxt(display, f"{eye_r:.0f}%", lp+134, row, (0, 200, 220), scale=0.38)
            row += 16
            _divider(display, px, row, pw)
            row += 10

            _stxt(display, f"POS ({cx},{cy})  SIZE {w}×{h}",
                  lp, row, DIM, scale=0.38)

            # Emotion bars (right of face)
            bx = x + w + 12
            if bx + 244 < W:
                bph = len(EMOTIONS) * (10 + 4) + 12
                _panel(display, bx-6, y-4, 250, bph, border=(52, 58, 72))
                _emotion_bars(display, scores, bx, y, bw=108)

        # ── Timeline ──────────────────────────────────────────────────────────
        if face_states:
            _timeline(display, next(iter(face_states.values())).timeline,
                      0, H - 42, W)

        # ── HUD ───────────────────────────────────────────────────────────────
        _panel(display, 0, 0, W, 36, alpha=0.90, border=(42, 48, 60))
        known_names = known.all_names()
        known_str   = f"  KNOWN: {', '.join(known_names)}" if known_names else ""
        _txt(display,
             f"  FACE TRACKER v4"
             f"   FACES: {len(boxes)}"
             f"   FPS: {fps:4.1f}"
             f"   FRAME: {frame_n:05d}"
             f"{known_str}"
             f"   [ R ] RESET   [ Q ] QUIT",
             8, 24, ACCENT, scale=0.55, thick=1)

        # ── Voice-capture overlay (big centre badge) ──────────────────────────
        vphase = voice_in.status["phase"]
        if vphase in ("listening", "thinking", "done", "error"):
            # Auto-clear "done"/"error" after a short display window
            if vphase in ("done", "error") \
               and now - voice_in.status.get("ends_at", now) > 2.5:
                voice_in.status["phase"] = "idle"
            else:
                bw, bh = 520, 160
                bx, by = (W - bw) // 2, (H - bh) // 2 - 40
                ov = display.copy()
                bgc = {"listening": (0, 0, 200),
                       "thinking":  (60, 60, 60),
                       "done":      (0, 140, 0),
                       "error":     (40, 40, 120)}.get(vphase, (40, 40, 40))
                cv2.rectangle(ov, (bx, by), (bx+bw, by+bh), bgc, -1)
                cv2.addWeighted(ov, 0.78, display, 0.22, 0, display)
                cv2.rectangle(display, (bx, by), (bx+bw, by+bh),
                              (255, 255, 255), 2)

                if vphase == "listening":
                    secs = max(0.0, voice_in.status["ends_at"] - now)
                    dot  = "●" if (frame_n // 8) % 2 == 0 else "○"
                    _txt(display, f"{dot}  RECORDING",
                         bx + 130, by + 60, (255, 255, 255), scale=1.1, thick=2)
                    _txt(display, f"Say your name now... {secs:3.1f}s",
                         bx + 90, by + 115, (255, 255, 255), scale=0.7, thick=1)
                elif vphase == "thinking":
                    _txt(display, "TRANSCRIBING...",
                         bx + 110, by + 60, (255, 255, 255), scale=1.1, thick=2)
                    _txt(display, "Converting speech to text via Google API",
                         bx + 55, by + 115, (220, 220, 220), scale=0.55, thick=1)
                elif vphase == "done":
                    _txt(display, "HEARD:",
                         bx + 30, by + 55, (255, 255, 255), scale=0.7, thick=1)
                    _txt(display, f'"{voice_in.status["text"]}"',
                         bx + 30, by + 115, (255, 255, 255), scale=1.0, thick=2)
                else:   # error
                    _txt(display, "VOICE ERROR",
                         bx + 130, by + 60, (255, 255, 255), scale=1.0, thick=2)
                    _txt(display, voice_in.status.get("error", ""),
                         bx + 30, by + 115, (240, 200, 200), scale=0.6, thick=1)

        # ── Reset-confirmation overlay (after first 'R' press) ────────────────
        if now < reset_pending_until:
            secs = reset_pending_until - now
            bw, bh = 560, 130
            bx, by = (W - bw) // 2, H - bh - 70
            ov = display.copy()
            cv2.rectangle(ov, (bx, by), (bx+bw, by+bh), (0, 60, 180), -1)
            cv2.addWeighted(ov, 0.82, display, 0.18, 0, display)
            cv2.rectangle(display, (bx, by), (bx+bw, by+bh), (255, 255, 255), 2)
            _txt(display, "PRESS  R  AGAIN  TO  CONFIRM  RESET",
                 bx + 30, by + 55, (255, 255, 255), scale=0.75, thick=2)
            _txt(display, f"Any other key cancels   ({secs:.1f}s)",
                 bx + 30, by + 100, (220, 220, 220), scale=0.55, thick=1)

        # Screen centre guide
        cv2.line(display, (W//2-14, H//2), (W//2+14, H//2), (48, 52, 62), 1)
        cv2.line(display, (W//2, H//2-14), (W//2, H//2+14), (48, 52, 62), 1)

        cv2.imshow("Face Tracker v4  —  Voice + Recognition", display)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            if now < reset_pending_until:
                # Second press within window → confirm
                n = known.clear()
                recog.clear()
                last_greeted.clear()
                recent_asks.clear()
                unknown_streak.clear()
                speaker.say(f"All {n} known faces have been reset."
                            if n else "There were no faces to reset.",
                            interrupt=True)
                reset_pending_until = 0.0
            else:
                # First press → arm
                reset_pending_until = now + 3.0
                speaker.say("Press R again to confirm reset.", interrupt=True)
        elif key != 255 and now < reset_pending_until:
            # Any other key cancels a pending reset
            reset_pending_until = 0.0

    cap.release()
    cv2.destroyAllWindows()
    speaker.stop()
    landmarker.close()
    print("Tracker stopped.")


if __name__ == "__main__":
    main()
