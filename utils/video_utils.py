import cv2

def read_video(video_path, override_fps=None):
    """
    Reads frames from video_path. If override_fps is set and lower than the
    native FPS, only returns frames sampled at ~override_fps.
    """
    cap = cv2.VideoCapture(video_path)
    print(f"Native fps is: {cap.get(cv2.CAP_PROP_FPS)}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = f"{width}x{height}"

    frames = []
    frame_idx = 0

    if override_fps and override_fps < native_fps:
        # time‐based downsampling
        next_time = 0.0
        time_step = 1.0 / override_fps
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = frame_idx / native_fps
            if current_time >= next_time:
                frames.append(frame)
                next_time += time_step
            frame_idx += 1
    else:
        # no override or override >= native: read everything
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

    cap.release()
    
    # If override_fps is set, adjust fps accordingly
    fps = override_fps if override_fps else native_fps

    return frames, fps, width, height, resolution

def save_video(output_video_frames, output_video_path, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    h, w = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
