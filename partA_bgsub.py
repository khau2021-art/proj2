import cv2, os, numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main(src=0, out_dir="data/output", write=True):
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise SystemExit("Cannot open video/camera")

    # GMM background model
    bgs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

    # Video writers (optional)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    bg_writer = cv2.VideoWriter(os.path.join(out_dir,"A_background_plate.mp4"), fourcc, fps, (w,h))
    m_writer  = cv2.VideoWriter(os.path.join(out_dir,"A_foreground_mask.mp4"), fourcc, fps, (w,h), False)

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break

        fgmask = bgs.apply(frame)                            # 0=bg, 255=fg(+shadows ~127)
        _, binmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)  # binarize moving FG

        # Background plate estimate: use background image if available
        bgimg = bgs.getBackgroundImage()
        if bgimg is None: bgimg = frame

        # show
        cv2.imshow("A: frame", frame)
        cv2.imshow("A: fgmask", binmask)

        if write:
            bg_writer.write(bgimg)
            m_writer.write(binmask)

        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break

    cap.release()
    bg_writer.release(); m_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
