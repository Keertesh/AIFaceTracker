# AI Face Tracker

A real-time face tracker that detects faces, recognises people by name, greets
them by voice, and reads **14 distinct emotions** from facial expressions —
all in Python, with no TensorFlow required.

Runs on Apple Silicon Macs with Python 3.12+ (tested on 3.14).

---

## What it does

When you point a webcam at it, the tracker:

- Detects every face in the frame in real time
- Reads **14 emotions** from facial expressions (built on 52 ARKit blendshapes
  from MediaPipe FaceLandmarker)
- Estimates **age** and **gender** with InsightFace's `buffalo_s` ONNX model
- Computes **head pose** (yaw / pitch / roll) from the facial transformation
  matrix
- Recognises **previously-seen people** using 512-dimensional MobileFaceNet
  embeddings persisted to disk
- When it sees a new face, it shows an **on-screen prompt** ("PLEASE TELL ME
  YOUR NAME"), plays a beep, records 4 seconds of audio from the mic, runs
  speech-to-text via Google Web Speech API, and saves the name with the face
- When a known face returns, it speaks an **emotion-aware greeting** such as
  *"I can see you Keertesh! You look really happy today!"*

The whole UI is a single OpenCV window with a tactical-HUD overlay: face mesh
dots, animated corner brackets, head-pose compass, smile / eye-openness bars,
distance estimate, sorted emotion confidence bars, and a scrolling emotion
timeline at the bottom of the frame.

---

## Quick start

```bash
git clone git@github.com:Keertesh/AIFaceTracker.git
cd AIFaceTracker
pip3 install -r requirements.txt
python3 face_tracker.py
```

On the first run two model files auto-download to `~/.face_tracker/`:

- MediaPipe `face_landmarker.task` (~4 MB)
- InsightFace `buffalo_s` pack (~75 MB) → `~/.insightface/models/`

After that, everything runs locally except the speech-to-text call (which
hits Google's free Web Speech API).

---

## macOS permissions

The first run will trigger two system prompts:

| Permission | Why | Where to enable |
|---|---|---|
| Camera | Webcam capture | System Settings > Privacy & Security > Camera |
| Microphone | Voice name capture | System Settings > Privacy & Security > Microphone |

If model download fails with an SSL certificate error (common with
python.org Python on macOS), run once:

```bash
/Applications/Python\ 3.14/Install\ Certificates.command
```

---

## Controls

| Key | Action |
|---|---|
| **Q** | Quit |
| **R**, then **R** within 3 s | Wipe every known face from disk (irreversible) |
| Any other key during the reset window | Cancels the pending reset |
| **Ctrl+C** | Clean shutdown from terminal |

---

## The 14 emotions

Each emotion is grounded in specific facial action units, not a black-box
classifier — so you can read the source and see exactly why a face was
classified the way it was.

| Emotion | Key facial signals |
|---|---|
| `HAPPY` | Bilateral smile + Duchenne cheek squint |
| `EXCITED` | Big smile + wide eyes + open mouth (high arousal) |
| `AMUSED` | Modest smile + knowing orbital squint |
| `SAD` | Mouth corners down + inner brow raise + lip pull-down |
| `ANGRY` | Brow compression + orbital squint + frown |
| `DISGUSTED` | Nose sneer + frown + brow compression |
| `CONTEMPT` | *Asymmetric* unilateral sneer / half-smile |
| `BORED` | Partial lid droop (eyelids ~ half-closed) |
| `SURPRISED` | Brow raise + wide eyes + jaw drop (moderate) |
| `SHOCKED` | All three surprise AUs at extreme amplitude simultaneously |
| `FEAR` | Medial brow raise + scleral show + mouth stretch |
| `CONFUSED` | *Asymmetric* brow (one up, one down) + mild squint |
| `CONCENTRATING` | *Symmetric* brow compression + pressed lips |
| `NEUTRAL` | Residual when all others are low |

Three design decisions that make this work in practice:

- **CONTEMPT** and **CONFUSED** use *asymmetry* as their main discriminator
  (left/right blendshape difference), not magnitude.
- **SHOCKED** uses a threshold-product (`max(jaw-0.55,0) × max(eye-0.55,0)
  × max(brow-0.48,0) × 120`) so it only fires when all three surprise muscles
  are at extreme amplitude simultaneously, preventing overlap with strong
  `SURPRISED`.
- **BORED** uses `blink × (1 − blink) × 4.5` which peaks when eyes are half
  closed (≈ 0.5 blink), distinguishing droopy-tired from full-blink.

---

## Pipeline

```
camera frame ─┬─► OpenCV Haar cascade ─────────────► face bounding boxes
              │
              ├─► MediaPipe FaceLandmarker (every frame)
              │       ├─► 478 landmarks ───► face mesh overlay
              │       ├─► 52 ARKit blendshapes ──► 14-emotion scores
              │       └─► 4×4 transformation matrix ──► yaw/pitch/roll
              │
              └─► InsightFace buffalo_s (every 25 frames, in a thread)
                      ├─► age (int)
                      ├─► gender ('M' / 'F')
                      └─► normed_embedding (512-D, L2-normalised)
                                │
                                └──► cosine sim vs ~/.face_tracker/known_faces.pkl
                                         ├─► match → spoken emotion greeting
                                         └─► no match → 6-condition gate
                                                          → record 4 s name
                                                          → Google STT
                                                          → save embedding+name
```

The "ask for name" prompt only fires when **all six** of these conditions
hold simultaneously (so a half-face won't trigger it):

1. Face is ≥ 130×130 px (good embedding quality)
2. Face is ≥ 14 px from every frame edge (not edge-clipped)
3. `|yaw| ≤ 25°` and `|pitch| ≤ 25°` (roughly frontal)
4. Status has been `UNKNOWN` for ≥ 5 consecutive analysis cycles
5. ≥ 15 s have passed since the last skipped attempt
6. Embedding doesn't match anything asked in the previous 60 s

---

## Data persistence

Everything lives in `~/.face_tracker/` so you can inspect, back up, or
delete it independently of the repo:

```
~/.face_tracker/
├── face_landmarker.task              MediaPipe model (~4 MB)
├── known_faces.pkl                   pickle: [{name, embedding(512-D)}, …]
└── audio/
    ├── name_1716534720.wav           every captured name recording
    └── name_1716534720.txt           paired transcript from Google STT
```

Press **R, R** to wipe `known_faces.pkl` without touching the audio archive.

---

## Tech stack

| Library | Purpose |
|---|---|
| OpenCV 4.13 | Webcam capture, Haar face detection, HUD rendering |
| MediaPipe 0.10.35 | Face landmarks, blendshapes, head-pose matrix |
| InsightFace 1.0 + ONNX Runtime | Age, gender, 512-D recognition embedding |
| SpeechRecognition 3.16 | Wraps Google Web Speech API for STT |
| sounddevice | Microphone capture (no PyAudio needed) |
| macOS `say` (built-in) | Non-blocking text-to-speech |
| macOS `afplay` (built-in) | System-sound beep before recording |

No TensorFlow / PyTorch / Keras dependency — important because TF doesn't
have wheels for Python 3.14 yet.

---

## Limitations

- macOS-only TTS (uses `say`). On other OSes the spoken greetings won't
  fire, but everything else works.
- Speech-to-text needs internet (Google Web Speech API). On failure the
  app falls back to typing the name in the terminal.
- Single-camera, single-process. No GPU acceleration is configured for
  ONNX Runtime — runs comfortably on a M2 CPU at ~25 fps.

---

## Roadmap

- [ ] Whisper-based offline STT as an internet-free alternative
- [ ] Per-person greeting personalisation
- [ ] Export emotion timeline as CSV / chart
- [ ] Cross-platform TTS (pyttsx3 fallback)

---

## Acknowledgments

- **MediaPipe Tasks** for the face landmarker bundle and the ARKit
  blendshape topology
- **InsightFace** for the buffalo_s ONNX model pack
  (`det_500m` + `genderage` + `w600k_mbf` + `2d106det`)
- The ARKit `ARFaceAnchor` blendshape coefficient naming, which makes the
  AU → emotion mapping readable instead of opaque
