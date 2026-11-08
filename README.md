# Driver Fatigue & Distraction Monitor

A real-time driver monitoring system using a webcam. It detects drowsiness and distraction from facial landmarks and plays a different warning sound for each.

Built with **MediaPipe FaceLandmarker**, **OpenCV** and an optional small **Keras** classifier.

## Features

- Face mesh tracking (468+ landmarks) with an on/off overlay
- Automatic 3-second calibration to your normal eyes, mouth and head position
- **Fatigue detection:** eye aspect ratio (EAR), PERCLOS, blink rate, yawn detection (MAR), microsleep alert
- **Distraction detection:** head pose (pitch, yaw, roll) compared to your baseline, alert after 5 s of looking away
- Two distinct alert sounds, generated in code (no audio files needed)
- Two modes for the fatigue meter:
  - **Rule-based** (default, no training needed)
  - **ML-based** (3-class model: normal / distracted / fatigue)

## Project Structure

```
driver_monitor.py        # main app (webcam, detection, alerts, data logging)
features.py              # feature list shared by logger, trainer and monitor
alerts_audio.py          # generated alert sounds
train_fatigue_model.py   # trains the optional ML model
```

Not included in the repo: `data/` (recorded samples), `fatigue_model.keras` (trained model) and `face_landmarker.task` (downloaded automatically on first run).

## Installation

Python 3.9+ is recommended.

```bash
pip install opencv-python mediapipe numpy
pip install tensorflow    # only needed for the ML mode
```

## Usage

Run the monitor with your webcam:

```bash
python driver_monitor.py
```

Or on a video file:

```bash
python driver_monitor.py --video path/to/video.mp4
```

At start, look straight ahead with eyes open and mouth closed for 3 seconds while it calibrates.

### Keys

| Key | Action |
|-----|--------|
| `m` | Mesh on/off |
| `r` | Recalibrate |
| `s` | Sound on/off |
| `l` | Data logging on/off |
| `0` / `1` / `2` | Set log label (normal / distracted / fatigue) |
| `q` | Quit |

The text next to the fatigue bar shows which mode is active: `(rule)` or `(ML)`.

## Training the ML Model (Optional)

The app works without this. You only need it if you want the ML-based meter.

**1. Record data.** Use one label per run, 2-3 minutes each, and several sessions per label:

```bash
python driver_monitor.py --log --label 0   # normal driving
python driver_monitor.py --log --label 1   # distracted (looking away, phone, mirrors)
python driver_monitor.py --log --label 2   # fatigue (yawns, slow blinks, eyes closing 1+ s)
```

Rows are saved to `data/features.csv`.

**2. Train:**

```bash
python train_fatigue_model.py
```

This creates `fatigue_model.keras`. The next time you start the monitor, it loads the model automatically and the bar shows `(ML)`.

To go back to rule-based mode, delete or rename `fatigue_model.keras`.

## Overfitting Caution

The model in this project **overfitted** on the data we recorded, so no dataset or trained model is included. If you train your own, keep this in mind:

- The validation split uses the last 20% of each session, so validation rows come from the same sessions as the training rows. **The reported accuracy is optimistic**, not a real test score.
- Data from one person, one camera, one room and one lighting setup will not generalize to other people or conditions.
- The model is small and the dataset is easy to memorize. Perfect or near-perfect validation accuracy is a warning sign, not a success.
- Acted drowsiness is not real drowsiness.

To get more trustworthy results:

- Record many sessions, with different people, lighting, camera distances and glasses on/off
- Test on whole sessions or people that were never used for training
- Check the confusion matrix, not just accuracy
- Do not rely on this project as a safety system for real driving

## Tuning

Thresholds are at the top of `driver_monitor.py`, for example `DISTRACT_SEC`, `EAR_RATIO`, `YAWN_DELTA`, `MICROSLEEP_SEC` and `PERCLOS_FULL`. If the rule-based meter feels too slow to rise, lower `PERCLOS_FULL`.

## Notes

- Alert sounds: Windows uses `winsound`, macOS uses `afplay`, Linux uses `paplay` or `aplay` (falls back to the terminal bell).
- This is a student/research project and **not** a certified safety product.

## License

Add your license here (for example, MIT).
