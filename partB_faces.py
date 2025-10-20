# partB_faces.py — Side-by-side comparison: LEFT=Haar (Viola–Jones), RIGHT=HOG (dlib)
# - High-res capture request (1080p -> fallback 720p)
# - OS-specific backend; MJPG on Win/Linux to avoid slow YUY2 conversion
# - Detect on downscaled copy; reuse detections every N frames
# - LEFT pane shows ONLY Haar boxes; RIGHT pane shows ONLY HOG boxes
# - Auto-downloads Haar cascade if missing; HOG pane warns if dlib unavailable

import os, sys, time, platform, argparse, urllib.request
import cv2

# Optional dlib (HOG)
try:
    import dlib
    DLIB_OK = True
except Exception as e:
    print(f"[WARN] dlib not available ({e}). HOG pane will show a warning.")
    DLIB_OK = False

HAAR_FILENAME = "haarcascade_frontalface_default.xml"
HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/" + HAAR_FILENAME

def fourcc_to_str(fourcc_int: int) -> str:
    return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

def ensure_haar() -> str:
    """Find or download haarcascade_frontalface_default.xml."""
    cv2_dir = os.path.dirname(cv2.__file__)
    pkg_path = os.path.join(cv2_dir, "data", HAAR_FILENAME)
    local_path = os.path.join(os.getcwd(), HAAR_FILENAME)
    if os.path.exists(pkg_path):
        return pkg_path
    if os.path.exists(local_path):
        return local_path
    print("[INFO] Haar cascade not found; downloading from OpenCV GitHub…")
    urllib.request.urlretrieve(HAAR_URL, local_path)
    print(f"[INFO] Saved to {local_path}")
    return local_path

def detect_haar(img_bgr, clf):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # mild contrast boost helps Haar
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def detect_hog(img_bgr, hogdet):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 0 = no upsample (faster). Use 1 if you need more recall and can afford the speed.
    rects = hogdet(gray, 0)
    return [(int(r.left()), int(r.top()), int(r.width()), int(r.height())) for r in rects]

def open_capture(src: int, want_w: int, want_h: int, want_fps: int):
    """Open camera with the best backend per OS and request MJPG when helpful."""
    system = platform.system().lower()
    if "windows" in system:
        backend = cv2.CAP_DSHOW
    elif "darwin" in system or "mac" in system:
        backend = cv2.CAP_AVFOUNDATION
    else:
        backend = cv2.CAP_V4L2

    cap = cv2.VideoCapture(src, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit("❌ Cannot open camera.")

    # Prefer MJPG on Win/Linux to avoid expensive YUY2 conversion
    if backend in (cv2.CAP_DSHOW, cv2.CAP_V4L2):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Try 1080p, then fallback to 720p if it doesn't stick
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
    cap.set(cv2.CAP_PROP_FPS, want_fps)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if h < 900 and want_h >= 1080:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, want_fps)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    print(f"[INFO] Capture: {w}x{h} @ {fps:.0f} FPS  FOURCC='{fourcc or '----'}'  Backend={backend}")
    return cap

def main():
    ap = argparse.ArgumentParser(description="Side-by-side Face Detection: LEFT=Haar, RIGHT=HOG")
    ap.add_argument("--src", type=int, default=0, help="Camera index")
    ap.add_argument("--w", type=int, default=1920, help="Request width (try 1920 -> 1280)")
    ap.add_argument("--h", type=int, default=1080, help="Request height (try 1080 -> 720)")
    ap.add_argument("--fps", type=int, default=30, help="Request FPS")
    ap.add_argument("--detect_w", type=int, default=640, help="Downscale width used for detection")
    ap.add_argument("--every", type=int, default=3, help="Run detectors every N frames (reuse between)")
    args = ap.parse_args()

    cap = open_capture(args.src, args.w, args.h, args.fps)

    # Resizable window
    cv2.namedWindow("Part B — Haar (left) vs HOG (right)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Part B — Haar (left) vs HOG (right)", 1280, 720)

    # Haar cascade
    haar_path = ensure_haar()
    haar = cv2.CascadeClassifier(haar_path)
    if haar.empty():
        sys.exit(f"❌ Failed to load Haar cascade from {haar_path}")

    # HOG detector (optional)
    hogdet = dlib.get_frontal_face_detector() if DLIB_OK else None

    print("Keys: q=quit | +/- = detection width | [ / ] = detect every N")
    frames, t0 = 0, time.time()
    boxes_haar_small = []
    boxes_hog_small  = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        # Downscale for detection
        h0, w0 = frame.shape[:2]
        scale = args.detect_w / float(w0)
        det_h = max(1, int(h0 * scale))
        small = cv2.resize(frame, (args.detect_w, det_h), interpolation=cv2.INTER_LINEAR)

        # Run detectors every N frames; reuse cached boxes otherwise
        if frames % args.every == 1:
            boxes_haar_small = detect_haar(small, haar)
            if DLIB_OK:
                boxes_hog_small = detect_hog(small, hogdet)
            else:
                boxes_hog_small = []

        # Build LEFT (Haar-only) pane
        left = frame.copy()
        for (x, y, w, h) in boxes_haar_small:
            X, Y, W, H = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
            cv2.rectangle(left, (X, Y), (X + W, Y + H), (255, 0, 0), 2)  # blue
        cv2.putText(left, "Haar (Viola–Jones)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Build RIGHT (HOG-only) pane
        right = frame.copy()
        if DLIB_OK:
            for (x, y, w, h) in boxes_hog_small:
                X, Y, W, H = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
                cv2.rectangle(right, (X, Y), (X + W, Y + H), (0, 255, 0), 2)  # green
            cv2.putText(right, "HOG (dlib)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(right, "HOG (dlib) unavailable", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Compose side-by-side (exactly as the project requires)
        split = cv2.hconcat([left, right])

        # Light FPS overlay (on the split image)
        if frames % 30 == 0:
            fps_live = frames / (time.time() - t0)
            cv2.putText(split, f"{fps_live:.1f} FPS  |  det_w={args.detect_w}px  every={args.every}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)

        cv2.imshow("Part B — Haar (left) vs HOG (right)", split)

        # Keys
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
        if k in (ord('+'), ord('=')):
            args.detect_w = min(960, args.detect_w + 40)   # increase detect resolution
        if k in (ord('-'), ord('_')):
            args.detect_w = max(320, args.detect_w - 40)   # decrease detect resolution
        if k == ord('['):
            args.every = max(1, args.every - 1)            # detect more often
        if k == ord(']'):
            args.every = min(10, args.every + 1)           # detect less often

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
