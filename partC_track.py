# partC_track.py — Real-time feature tracking (Shi–Tomasi + Lucas–Kanade) with auto re-detect
# Flow:
#   1) Detect a face (Haar by default, optional HOG) to seed an ROI
#   2) Extract good feature corners inside ROI
#   3) Track features with cv2.calcOpticalFlowPyrLK
#   4) If features fall below threshold, re-detect ROI
#   5) Draw a red rectangle for the tracked object; optional corner dots

import os, sys, time, platform, argparse
import cv2
import numpy as np

# --- Optional dlib (HOG) ---
try:
    import dlib
    DLIB_OK = True
except Exception as e:
    print(f"[WARN] dlib not available ({e}). HOG disabled.")
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
    if os.path.exists(pkg_path): return pkg_path
    if os.path.exists(local_path): return local_path
    # Fall back: try to download (requires internet)
    try:
        import urllib.request
        print("[INFO] Haar not found; downloading…")
        urllib.request.urlretrieve(HAAR_URL, local_path)
        print(f"[INFO] Saved to {local_path}")
        return local_path
    except Exception:
        raise FileNotFoundError(
            f"Cannot find {HAAR_FILENAME}. Place it next to this script or in cv2/data."
        )

def open_capture(src: int, want_w: int, want_h: int, want_fps: int):
    """Open camera with a good backend and try for HD."""
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

    # Prefer MJPG on Win/Linux to avoid slow YUY2 conversions
    if backend in (cv2.CAP_DSHOW, cv2.CAP_V4L2):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
    cap.set(cv2.CAP_PROP_FPS, want_fps)

    # Fallback to 720p if 1080p didn't stick
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

def detect_face_haar(img_bgr, clf):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    # return biggest face (x,y,w,h) or None
    if len(faces) == 0: return None
    return max(faces, key=lambda b: b[2]*b[3])

def detect_face_hog(img_bgr, hogdet):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rects = hogdet(gray, 0)  # 0 = speed; use 1 for more recall if needed
    if len(rects) == 0: return None
    # convert to (x,y,w,h) and pick largest
    boxes = [(r.left(), r.top(), r.width(), r.height()) for r in rects]
    return max(boxes, key=lambda b: b[2]*b[3])

def good_corners(gray, roi, maxN=150):
    """Return Shi–Tomasi corners inside roi=(x,y,w,h) in absolute coords."""
    x,y,w,h = roi
    x = max(0, x); y = max(0, y)
    sub = gray[y:y+h, x:x+w]
    if sub.size == 0: return None
    pts = cv2.goodFeaturesToTrack(sub, maxCorners=maxN, qualityLevel=0.01, minDistance=5, blockSize=7)
    if pts is None: return None
    pts[:,0,0] += x; pts[:,0,1] += y
    return pts

def main():
    ap = argparse.ArgumentParser(description="Part C — Feature Tracking with LK and Auto Re-detect")
    ap.add_argument("--src", type=int, default=0, help="Camera index")
    ap.add_argument("--w", type=int, default=1920, help="Request width (try 1920 → 1280)")
    ap.add_argument("--h", type=int, default=1080, help="Request height (try 1080 → 720)")
    ap.add_argument("--fps", type=int, default=30, help="Request FPS")
    ap.add_argument("--detect_w", type=int, default=640, help="Downscale width used for detection")
    ap.add_argument("--every", type=int, default=3, help="Run detector every N frames while searching")
    ap.add_argument("--min_keep", type=float, default=0.5, help="Re-detect when features < min_keep * initial")
    args = ap.parse_args()

    cap = open_capture(args.src, args.w, args.h, args.fps)

    # UI
    cv2.namedWindow("Part C — Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Part C — Tracking", 1280, 720)
    print("Keys: q=quit | h=toggle Haar/HOG | r=force re-detect | p=toggle points | +/- min_keep | [ / ] detect-every")

    # Detectors
    haar_path = ensure_haar()
    haar = cv2.CascadeClassifier(haar_path)
    if haar.empty():
        sys.exit(f"❌ Failed to load Haar from {haar_path}")
    hogdet = dlib.get_frontal_face_detector() if DLIB_OK else None

    use_haar_only = not DLIB_OK   # if dlib missing, stick to Haar
    use_haar = True               # default detector when searching
    show_points = True

    # LK params
    lk = dict(winSize=(21,21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Tracker state
    tracking = False
    prev_gray = None
    pts_prev = None
    bbox = None
    init_pts_count = 0
    frames, t0 = 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok: break
        frames += 1

        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not tracking:
            # Downscale for faster detector pass
            h0,w0 = frame.shape[:2]
            scale = args.detect_w / float(w0)
            small = cv2.resize(frame, (args.detect_w, int(h0*scale)), interpolation=cv2.INTER_LINEAR)

            # Run detector every N frames while not tracking
            do_detect_now = (frames % args.every == 1)
            if do_detect_now:
                if use_haar or use_haar_only:
                    box_small = detect_face_haar(small, haar)
                else:
                    box_small = detect_face_hog(small, hogdet)

                if box_small is not None:
                    x,y,w,h = box_small
                    # scale back to full-res coords
                    bbox = (int(x/scale), int(y/scale), int(w/scale), int(h/scale))

                    # Seed feature points inside bbox
                    pts_prev = good_corners(gray, bbox, maxN=200)
                    if pts_prev is not None:
                        init_pts_count = len(pts_prev)
                        prev_gray = gray.copy()
                        tracking = True

            # Status overlay while searching
            cv2.putText(vis, f"Searching... Detector={'Haar' if (use_haar or use_haar_only) else 'HOG'}  "
                             f"detect_w={args.detect_w}px every={args.every}",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        else:
            # Track with LK
            pts_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None, **lk)

            good_new = pts_next[st==1] if pts_next is not None else np.empty((0,2))
            good_old = pts_prev[st==1] if pts_prev is not None else np.empty((0,2))

            # If we have tracked points, update bbox by median displacement
            if len(good_new) >= 3:
                disp = np.median(good_new - good_old, axis=0)
                dx, dy = float(disp[0]), float(disp[1])
                x,y,w,h = bbox
                bbox = (int(x+dx), int(y+dy), w, h)

                # Re-seed fresh corners in new bbox
                pts_prev = good_corners(gray, bbox, maxN=200)
                prev_gray = gray.copy()
            else:
                # Not enough tracked points — force re-detect
                tracking = False
                pts_prev = None
                prev_gray = None

            # Re-detect if feature count drops too low
            if tracking and (pts_prev is None or len(pts_prev) < max(3, int(init_pts_count * args.min_keep))):
                tracking = False
                pts_prev = None
                prev_gray = None

            # Draw ROI and (optionally) points
            if bbox is not None:
                x,y,w,h = bbox
                cv2.rectangle(vis, (x,y), (x+w,y+h), (0,0,255), 2)  # red box
                if show_points and pts_prev is not None:
                    for p in pts_prev.reshape(-1,2):
                        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0,255,0), -1)

            cv2.putText(vis, f"Tracking ({len(pts_prev) if pts_prev is not None else 0} pts)  "
                             f"min_keep={args.min_keep:.2f}",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # FPS overlay
        if frames % 30 == 0:
            fps_live = frames / (time.time() - t0)
            cv2.putText(vis, f"{fps_live:.1f} FPS", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50), 2)

        cv2.imshow("Part C — Tracking", vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break
        if k == ord('h') and not use_haar_only: use_haar = not use_haar
        if k == ord('r'):
            tracking = False; pts_prev=None; prev_gray=None; bbox=None  # force re-detect
        if k == ord('p'):
            show_points = not show_points
        if k in (ord('+'), ord('=')):
            args.min_keep = min(0.95, args.min_keep + 0.05)
        if k in (ord('-'), ord('_')):
            args.min_keep = max(0.10, args.min_keep - 0.05)
        if k == ord('['):
            args.every = max(1, args.every - 1)
        if k == ord(']'):
            args.every = min(10, args.every + 1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
