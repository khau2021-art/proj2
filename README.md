Project 2 - Real-Time Face Detection and Tracking (CPET-347)

Full Pipeline: Capture → Detect → Track → Integrate
Environment: Anaconda (Python 3.11), OpenCV, dlib (optional)

Setup

Create and activate your conda environment:
conda create -n proj2 python=3.11 -y
conda activate proj2

Install dependencies:
pip install opencv-python dlib numpy
If dlib fails to install, Haar detection will still run (HOG pane disabled).

Folder Layout

proj2/
	data
		input
		output
	models
	out
	haarcascade_frontalface_default.xml
	partA_bgsub.py
	partB_faces.py
	partC_track.py
	partD_integrated.py
	README.md

Part A – Foreground and Background Segmentation

Goal:
Use a background model (MOG2) to separate moving foreground objects from static background.

Run:
python partA_bgsub.py

Controls:
q = quit
Left window = live feed
Right window = foreground mask

Demonstrates motion-based segmentation versus static background.

Part B – Face Detection Comparison (Haar vs HOG)

Goal:
Compare two classical face detectors side-by-side.

Run:
python partB_faces.py

Behavior:
Left pane uses Haar cascade (Viola–Jones)
Right pane uses HOG + SVM (dlib)
Both run live at about 15 FPS (1080p preferred, 720p fallback)
Haar boxes are blue, HOG boxes are green

Controls:
q = quit
[ and ] = adjust detection interval

and - = change downscale width

Displays FPS and detection settings in top-left.
Haar runs faster but less accurate.
HOG is slower but more robust to pose and lighting.

Part C – Face Tracking (Lucas–Kanade Optical Flow)

Goal:
Track a detected face across frames without re-detecting every frame.

Run:
python partC_track.py

Process:

Haar detector finds initial face region.

Good Features to Track corners are selected within that ROI.

Lucas–Kanade optical flow tracks these points across frames.

Median motion updates the bounding box position.

Modes:
Baseline – intentional drift to show error accumulation.
Stabilized – uses forward/backward validation and RANSAC affine to remove drift.

Controls:
f = toggle baseline/stabilized
r = re-detect face
p = toggle corner points
q = quit

Start in baseline mode to show drifting box.
Press f to switch to stabilized mode for steady tracking.

Part D – Integrated Real-Time Pipeline

Goal:
Combine all previous stages into one integrated application.

Run:
python partD_integrated.py --src 0 --w 1920 --h 1080 --fps 30

Modes:
1 = Part A (FG/BG)
2 = Part B (Haar vs HOG)
3 = Part C (Haar + LK tracking)

Global Keys:
q = quit
u = toggle undistortion
v = record on/off (MP4)
s = save snapshot

Detection Controls:
[ and ] = change detection interval

and - = change detection scale

Tracking Controls:
r = re-detect face
p = toggle feature points
f = toggle drift-fix mode

Outputs:
Displays FPS and mode labels in top-left.
Snapshots and recordings saved to /out/.
Automatically downloads Haar cascade if missing.

Expected Results and Discussion

Part A:
Foreground mask reacts to motion and slowly clears for stationary objects.

Part B:
Haar runs fast but is less robust to tilt or lighting.
HOG detects more reliably under pose and brightness changes.

Part C:
LK tracking runs smoothly at 15–30 FPS.
Baseline drifts over time, stabilized mode maintains position.

Part D:
Integrated application demonstrates a full real-time vision pipeline using classical computer vision techniques.

End of Project 2 README