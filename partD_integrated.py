# partD_integrated.py — CPET347 Project 2: Integrated App
# Capture → (optional) Undistort → (A) FG/BG → (B) Faces (Haar vs HOG) → (C) LK Tracking (Haar)
# Keys:
#   Global: q=quit, u=toggle undistort, v=record on/off, s=snapshot
#   Modes:  1=Part A (FG/BG), 2=Part B (Haar left vs HOG right), 3=Part C (Haar-track)
#   Detect: [ / ] = detect every N frames,  + / - = detect downscale width
#   Track : r=re-detect, p=points on/off, f=toggle drift-fix (baseline vs stabilized)

import os, sys, time, platform, argparse, urllib.request, datetime
import cv2
import numpy as np

# -------- Optional dlib (HOG) --------
try:
    import dlib
    DLIB_OK = True
except Exception as e:
    print(f"[WARN] dlib not available ({e}). HOG (right pane in Part B) will be disabled.")
    DLIB_OK = False

HAAR_FILENAME = "haarcascade_frontalface_default.xml"
HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/" + HAAR_FILENAME

def fourcc_to_str(fourcc_int: int) -> str:
    return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

def ensure_haar() -> str:
    cv2_dir = os.path.dirname(cv2.__file__)
    pkg_path = os.path.join(cv2_dir, "data", HAAR_FILENAME)
    local_path = os.path.join(os.getcwd(), HAAR_FILENAME)
    if os.path.exists(pkg_path): return pkg_path
    if os.path.exists(local_path): return local_path
    print("[INFO] Haar cascade not found; downloading…")
    urllib.request.urlretrieve(HAAR_URL, local_path)
    print(f"[INFO] Saved to {local_path}")
    return local_path

def open_capture(src: int, want_w: int, want_h: int, want_fps: int):
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

    if backend in (cv2.CAP_DSHOW, cv2.CAP_V4L2):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
    cap.set(cv2.CAP_PROP_FPS, want_fps)

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

# ---------- Part A: FG/BG ----------
def make_bg_model():
    # MOG2: shadows=True makes shadows gray (not white)
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# ---------- Part B: Face detection ----------
def detect_haar(img_bgr, clf):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def detect_hog(img_bgr, hogdet, upsample=0):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rects = hogdet(gray, upsample)
    return [(int(r.left()), int(r.top()), int(r.width()), int(r.height())) for r in rects]

# ---------- Part C: LK tracking (Haar-seeded) ----------
def good_corners(gray, roi, maxN=200):
    x,y,w,h = roi
    x = max(0, x); y = max(0, y)
    sub = gray[y:y+h, x:x+w]
    if sub.size == 0: return None
    pts = cv2.goodFeaturesToTrack(sub, maxCorners=maxN, qualityLevel=0.01, minDistance=5, blockSize=7)
    if pts is None: return None
    pts[:,0,0] += x; pts[:,0,1] += y
    return pts

def bbox_to_quad(b):
    x,y,w,h = b
    return np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])

def quad_to_bbox(Q):
    xs, ys = Q[:,0], Q[:,1]
    x, y = xs.min(), ys.min()
    return (int(round(x)), int(round(y)), int(round(xs.max()-x)), int(round(ys.max()-y)))

def apply_affine_to_bbox(bbox, A2x3, img_shape):
    Q = bbox_to_quad(bbox).reshape(-1,1,2)
    Qt = cv2.transform(Q, A2x3).reshape(-1,2)
    h, w = img_shape[:2]
    Qt[:,0] = np.clip(Qt[:,0], 0, w-1)
    Qt[:,1] = np.clip(Qt[:,1], 0, h-1)
    return quad_to_bbox(Qt)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Part D — Integrated App")
    ap.add_argument("--src", type=int, default=0, help="Camera index")
    ap.add_argument("--w", type=int, default=1920, help="Request width (try 1920 → 1280)")
    ap.add_argument("--h", type=int, default=1080, help="Request height (try 1080 → 720)")
    ap.add_argument("--fps", type=int, default=30, help="Request FPS")
    ap.add_argument("--detect_w", type=int, default=640, help="Downscale width for detection")
    ap.add_argument("--every", type=int, default=3, help="Run detectors every N frames (reuse between)")
    ap.add_argument("--hog_upsample", type=int, default=0, help="HOG upsample (0 fast, 1 better recall)")
    ap.add_argument("--no-undistort", action="store_true", help="Disable undistortion stage")
    args = ap.parse_args()

    os.makedirs("out", exist_ok=True)

    cap = open_capture(args.src, args.w, args.h, args.fps)

    # Windows
    cv2.namedWindow("Part D — Integrated", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Part D — Integrated", 1280, 720)

    # Undistort maps (identity if disabled)
    if args.no_undistort:
        map1 = map2 = None
    else:
        # Use camera intrinsics if available; otherwise identity (here we use identity “same K” trick).
        # This keeps the interface consistent without requiring a prior calibration file.
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        K = np.array([[w0,   0, w0/2],
                      [  0, h0, h0/2],
                      [  0,   0,   1]], dtype=np.float32)
        dist = np.zeros(5, dtype=np.float32)
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w0, h0), cv2.CV_16SC2)

    # Part A
    bg = make_bg_model()

    # Part B
    haar_path = ensure_haar()
    haar = cv2.CascadeClassifier(haar_path)
    if haar.empty():
        sys.exit(f"❌ Failed to load Haar cascade from {haar_path}")
    hogdet = dlib.get_frontal_face_detector() if DLIB_OK else None

    # Part C tracking state (Haar-only seeding)
    lk = dict(winSize=(21,21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    tracking = False
    prev_gray = None
    pts_prev = None
    bbox = None
    init_pts_count = 0
    show_points = True
    use_stabilized = False  # ‘f’ toggles baseline drift vs RANSAC-stabilized

    # Common
    mode = 2  # start at Part B per spec
    frames, t0 = 0, time.time()
    boxes_haar_small, boxes_hog_small = [], []
    record = False
    writer = None

    print("Modes: 1=Part A FG/BG | 2=Part B Haar vs HOG | 3=Part C LK Tracking (Haar)")
    print("Global: q quit | u undistort on/off | v record on/off | s snapshot")
    print("Detect: [ / ] every-N | + / - detect_w")
    print("Track : r re-detect | p points | f stabilized toggle")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frames += 1
        # Undistort if enabled
        if map1 is not None:
            frame_u = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        else:
            frame_u = frame

        h0, w0 = frame_u.shape[:2]
        vis = None

        # ---------- Part A: FG/BG ----------
        if mode == 1:
            fg = bg.apply(frame_u)
            # Optional: clean mask a bit
            fg_clean = cv2.medianBlur(fg, 5)
            bg_img = bg.getBackgroundImage()
            left = frame_u.copy()
            right = cv2.cvtColor(fg_clean, cv2.COLOR_GRAY2BGR)
            if bg_img is not None:
                # stack: top row live vs mask, bottom row background model
                top = cv2.hconcat([left, right])
                bot = cv2.hconcat([bg_img, bg_img])
                vis = cv2.vconcat([top, bot])
            else:
                vis = cv2.hconcat([left, right])
            cv2.putText(vis, "Part A — FG/BG (MOG2): Left=Live, Right=Mask; Bottom=Background Model",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # ---------- Part B: Haar left vs HOG right ----------
        elif mode == 2:
            # Downscale for detection
            scale = args.detect_w / float(w0)
            small = cv2.resize(frame_u, (args.detect_w, int(h0*scale)), interpolation=cv2.INTER_LINEAR)
            if frames % args.every == 1:
                boxes_haar_small = detect_haar(small, haar)
                boxes_hog_small  = detect_hog(small, hogdet, upsample=args.hog_upsample) if DLIB_OK else []

            left = frame_u.copy()
            for (x,y,w,h) in boxes_haar_small:
                X,Y,W,H = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                cv2.rectangle(left, (X,Y), (X+W,Y+H), (255,0,0), 2)
            cv2.putText(left, "Haar (Viola–Jones)", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            right = frame_u.copy()
            if DLIB_OK:
                for (x,y,w,h) in boxes_hog_small:
                    X,Y,W,H = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                    cv2.rectangle(right, (X,Y), (X+W,Y+H), (0,255,0), 2)
                cv2.putText(right, "HOG (dlib)", (10,25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            else:
                cv2.putText(right, "HOG (dlib) unavailable", (10,25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            vis = cv2.hconcat([left, right])

            # HUD
            if frames % 30 == 0:
                fps_live = frames / (time.time() - t0)
                cv2.putText(vis, f"{fps_live:.1f} FPS | det_w={args.detect_w}px every={args.every}",
                            (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50), 2)

        # ---------- Part C: LK tracking (Haar seed) ----------
        else:
            gray = cv2.cvtColor(frame_u, cv2.COLOR_BGR2GRAY)

            if not tracking:
                scale = args.detect_w / float(w0)
                small = cv2.resize(frame_u, (args.detect_w, int(h0*scale)), interpolation=cv2.INTER_LINEAR)
                if frames % args.every == 1:
                    faces = detect_haar(small, haar)
                    if faces:
                        # pick largest face
                        x,y,w,h = max(faces, key=lambda b: b[2]*b[3])
                        bbox = (int(x/scale), int(y/scale), int(w/scale), int(h/scale))
                        pts_prev = good_corners(gray, bbox, maxN=200)
                        if pts_prev is not None:
                            init_pts_count = len(pts_prev)
                            prev_gray = gray.copy()
                            tracking = True
                vis = frame_u.copy()
                cv2.putText(vis, f"Part C — Searching (Haar). det_w={args.detect_w}px every={args.every}",
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            else:
                vis = frame_u.copy()
                # Baseline (intentional drift) vs Stabilized (RANSAC)
                if not use_stabilized:
                    pts_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None,
                                                                 winSize=(21,21), maxLevel=3,
                                                                 criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
                    if pts_next is None or st is None:
                        tracking = False; pts_prev=None; prev_gray=None
                    else:
                        good_new = pts_next[st==1]
                        good_old = pts_prev[st==1]
                        if len(good_new) >= 3:
                            disp = np.median(good_new - good_old, axis=0)
                            dx, dy = float(disp[0]), float(disp[1])
                            x,y,w,h = bbox
                            bbox = (int(x+dx), int(y+dy), w, h)
                            # reseed corners
                            pts_prev = good_corners(gray, bbox, maxN=200)
                            prev_gray = gray.copy()
                        else:
                            tracking = False; pts_prev=None; prev_gray=None
                    mode_label = "Baseline (drift)"
                else:
                    pts_next, st_fwd, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None,
                                                                     winSize=(21,21), maxLevel=3,
                                                                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
                    if pts_next is None:
                        tracking = False; pts_prev=None; prev_gray=None
                    else:
                        pts_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, pts_next, None,
                                                                       winSize=(21,21), maxLevel=3,
                                                                       criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
                        fb_err = np.linalg.norm(pts_prev - pts_back, axis=2)
                        ok = (st_fwd.reshape(-1)==1) & (st_bwd.reshape(-1)==1) & (fb_err.reshape(-1) < 1.5)
                        if np.count_nonzero(ok) < 6:
                            tracking = False; pts_prev=None; prev_gray=None
                        else:
                            old = pts_prev.reshape(-1,2)[ok].astype(np.float32)
                            new = pts_next.reshape(-1,2)[ok].astype(np.float32)
                            A, inl = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC,
                                                                 ransacReprojThreshold=3.0, maxIters=2000,
                                                                 confidence=0.99, refineIters=10)
                            if A is None or inl is None or inl.sum() < 6:
                                tracking = False; pts_prev=None; prev_gray=None
                            else:
                                bbox = apply_affine_to_bbox(bbox, A, frame_u.shape)
                                pts_prev = good_corners(gray, bbox, maxN=200)
                                prev_gray = gray.copy()
                    mode_label = "Stabilized (RANSAC)"

                # Draw
                if bbox is not None:
                    x,y,w,h = bbox
                    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,0,255), 2)
                    if show_points and pts_prev is not None:
                        for p in pts_prev.reshape(-1,2):
                            cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0,255,0), -1)

                cv2.putText(vis, f"Part C — Tracking | {mode_label}", (10,25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # ---------- Common HUD ----------
        if vis is None:
            vis = frame_u
        if frames % 30 == 0:
            fps_live = frames / (time.time() - t0)
            cv2.putText(vis, f"{fps_live:.1f} FPS", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50), 2)

        # Recording
        if record and writer is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            outp = os.path.join("out", f"partD_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(outp, fourcc, max(10, int(cap.get(cv2.CAP_PROP_FPS))), (vis.shape[1], vis.shape[0]))
            print(f"[REC] Recording to {outp}")
        if writer is not None:
            writer.write(vis)

        cv2.imshow("Part D — Integrated", vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break
        # Mode switches
        if k == ord('1'): mode = 1
        if k == ord('2'): mode = 2
        if k == ord('3'): mode = 3
        # Global toggles
        if k == ord('u'):
            if map1 is None:
                # enable identity undistort maps (same K->K)
                w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                K = np.array([[w0,0,w0/2],[0,h0,h0/2],[0,0,1]], np.float32)
                dist = np.zeros(5, np.float32)
                map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w0,h0), cv2.CV_16SC2)
                print("[INFO] Undistort: enabled (identity model)")
            else:
                map1 = map2 = None
                print("[INFO] Undistort: disabled")
        if k == ord('v'):
            record = not record
            if not record and writer is not None:
                writer.release(); writer = None
                print("[REC] Stopped")
        if k == ord('s'):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("out", f"snap_{ts}.png")
            cv2.imwrite(path, vis); print(f"[SAVE] {path}")

        # Detection parameters
        if k in (ord('['),):
            args.every = max(1, args.every - 1)
        if k in (ord(']'),):
            args.every = min(10, args.every + 1)
        if k in (ord('+'), ord('=')):
            args.detect_w = min(960, args.detect_w + 40)
        if k in (ord('-'), ord('_')):
            args.detect_w = max(320, args.detect_w - 40)

        # Tracking toggles (Part C)
        if k == ord('r'):
            tracking = False; pts_prev=None; prev_gray=None; bbox=None
        if k == ord('p'):
            show_points = not show_points
        if k == ord('f'):
            use_stabilized = not use_stabilized

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
