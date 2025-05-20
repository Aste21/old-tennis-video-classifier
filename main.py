import os
import cv2
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

from utils.video_utils import read_video, save_video
from utils.bbox_utils import get_center_of_bbox
from utils.plot_utils import plot_ball_trajectory
from trackers import BallTracker
from court_line_detector import CourtLineDetector

from functools import lru_cache

import matplotlib.pyplot as plt


def main():
    # ─── Config ───────────────────────────────────────────────────────────────
    input_video_path = "input_videos/testmatch2_short.mp4"
    override_fps = None
    sample_rate_sec = 1.0
    min_segment_duration_s = 2.0

    # ─── Start total timer ────────────────────────────────────────────────────
    t_start_total = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ─── 1) Load video & FPS ─────────────────────────────────────────────────
    t0 = time.time()
    print("[1/8] Loading video and reading FPS…")
    video_frames, fps, width, height, resolution = read_video(
        input_video_path, override_fps
    )

    n_frames = len(video_frames)
    cap = cv2.VideoCapture(input_video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    fps = override_fps if override_fps else native_fps
    video_length_sec = n_frames / fps
    print(
        f"[1/8] Loaded {n_frames} frames at {fps:.2f} fps "
        f"(native {native_fps:.2f}), length {video_length_sec:.1f}s"
    )
    runtimes = {"load_video": time.time() - t0}

    # ─── 2) Coarse sampling for court presence ────────────────────────────────
    t0 = time.time()
    print("[2/8] Sampling for court presence…")
    court_model = CourtLineDetector("models/keypoints_model.pth")
    sample_step = max(1, int(fps * sample_rate_sec))
    sample_idxs = list(range(0, n_frames, sample_step))
    sample_flags = []
    for idx in sample_idxs:
        flag = court_model.is_court_frame(video_frames[idx])
        sample_flags.append(flag)
    print(f"[2/8] Completed {len(sample_idxs)} samples")
    runtimes["sampling"] = time.time() - t0

    # ─── 3) Collapse samples into coarse segments ───────────────────────────
    t0 = time.time()
    print("[3/8] Collapsing samples into coarse segments…")

    coarse_segs = []
    seg_start = None
    for idx, flag in zip(sample_idxs, sample_flags):
        if flag and seg_start is None:
            seg_start = idx
        elif not flag and seg_start is not None:
            seg_end = idx  # first FALSE after a run of TRUE
            if (seg_end - seg_start) / fps >= min_segment_duration_s:
                coarse_segs.append((seg_start, seg_end))
            seg_start = None
    if seg_start is not None:  # tail segment
        seg_end = sample_idxs[-1] + sample_step
        if (seg_end - seg_start) / fps >= min_segment_duration_s:
            coarse_segs.append((seg_start, seg_end))

    runtimes["segment_collapse"] = time.time() - t0
    print(f"[3/8] Coarse segments: {coarse_segs}")

    if not coarse_segs:
        print("[Done] No play segments found.")
        return

    # ─── 3b) Refine segment boundaries with log₂ search ────────────────────
    t0 = time.time()
    print("[3b] Refining segment boundaries…")

    @lru_cache(maxsize=None)
    def cached_is_court(idx: int) -> bool:
        """Memoised wrapper – the same frame is never scored twice."""
        return court_model.is_court_frame(video_frames[idx])

    def refine_edge(lo: int, hi: int, want_first_true: bool) -> int:
        """
        Binary-search between lo and hi (lo < hi) where the labels differ.
        Return the index of the first TRUE (if want_first_true) or
        the last TRUE (if not).
        """
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if cached_is_court(mid) ^ want_first_true:  # still on the FALSE side
                lo = mid
            else:
                hi = mid
        return hi if want_first_true else lo

    refined_segs = []
    for seg_start, seg_end in coarse_segs:
        # entry pair: FALSE → TRUE
        enter_false = max(0, seg_start - sample_step)
        enter_true = seg_start
        # exit pair: TRUE → FALSE
        exit_true = seg_end - sample_step
        exit_false = seg_end

        start_idx = refine_edge(enter_false, enter_true, want_first_true=True)
        end_idx = refine_edge(exit_true, exit_false, want_first_true=False)

        # keep semantics “end is exclusive” for the range() that follows later
        end_idx += 1
        if (end_idx - start_idx) / fps >= min_segment_duration_s:
            refined_segs.append((start_idx, end_idx))

    coarse_segs = refined_segs
    runtimes["segment_refine"] = time.time() - t0
    print(f"[3b] Refined segments: {coarse_segs}")

    # ─── 4) Expand to full-frame indices ───────────────────────────────────
    t0 = time.time()
    print("[4/8] Expanding segments to full-frame indices…")
    court_frame_indices = [list(range(start, end)) for start, end in coarse_segs]
    print(f"[4/8] Total court frames = {len(court_frame_indices)}")
    court_frames = [
        [video_frames[i] for i in segment] for segment in court_frame_indices
    ]
    runtimes["expand_frames"] = time.time() - t0

    print(f"coarse_seg = {coarse_segs}")
    print(f"court_frame_indices = {court_frame_indices}")
    # print(f"court_frames = {court_frames}")

    # ─── 5) Ball tracking ────────────────────────────────────────────────────
    t0 = time.time()
    print("[5/8] Running ball tracker on court frames…")
    ball_tracker = BallTracker(model_path="models/yolo5_last.pt")
    ball_detections = []
    for frame_segment in court_frames:
        raw_detections = ball_tracker.detect_frames(frame_segment)
        ball_detections.append(ball_tracker.interpolate_ball_positions(raw_detections))
    print(f"[5/8] Tracked ball on {len(ball_detections)} segments")
    runtimes["ball_tracking"] = time.time() - t0
    
    # ─── 5b) Save trajectory plots ──────────────────────────────────────────────
    print("[5b/8] Saving trajectory plots...")
    os.makedirs("output_data", exist_ok=True)
    court_dimensions = (width, height)  # From video metadata
    
    for seg_idx, (segment_detections, (start_frame, end_frame)) in enumerate(zip(ball_detections, coarse_segs), start=1):
        plot_filename = f"{ts}_play{seg_idx:02d}_trajectory.png"
        plot_path = os.path.join("output_data", plot_filename)
        
        plot_ball_trajectory(
            segment_detections,
            plot_path,
            seg_idx,
            court_dimensions=court_dimensions
        )
        print(f"Saved trajectory plot: {plot_path}")
    
    #! test of ball analysis
    ball_analysis = []
    for ball_segment in ball_detections:
        ball_analysis = ball_tracker.get_ball_shot_frames(ball_segment)
        
    print(f"Ball analysis: {ball_analysis}")
    """
    # ─── 6) Build rows + save one CSV per play ───────────────────────────────
    t0 = time.time()
    print("[6/8] Saving per-play CSV files…")

    # 6a  create a master list of ball positions --------------------------------
    rows = []
    for orig_idx, det in zip(court_frame_indices, ball_detections):
        bbox = det.get(1)
        if bbox:
            x_c, y_c = get_center_of_bbox(bbox)
        else:
            x_c, y_c = np.nan, np.nan
        rows.append(
            {
                "frame": orig_idx,
                "time_sec": orig_idx / fps,
                "x_px": x_c,
                "y_px": y_c,
            }
        )

    df_all = pd.DataFrame(rows)

    # 6b  write one CSV for every refined segment --------------------------------
    os.makedirs("output_data", exist_ok=True)
    for seg_idx, (start_f, end_f) in enumerate(coarse_segs, start=1):
        clip_df = df_all[
            (df_all["frame"] >= start_f) & (df_all["frame"] <= end_f)
        ].copy()

        csv_name = f"{ts}_play{seg_idx:02d}.csv"
        csv_path = os.path.join("output_data", csv_name)
        clip_df.to_csv(csv_path, index=False)
        print(f"play {seg_idx}: {csv_path}")

    runtimes["save_csv"] = time.time() - t0
    """
    # ─── 7) Annotate & save video ────────────────────────────────────────────
    t0 = time.time()
    print("[7/8] Annotating and saving video…")
    annotated = []

    # Precompute a lookup: {frame_number: detection_dict}
    frame_to_detections = {}
    for seg_idx, segment in enumerate(court_frame_indices):
        for frame_pos, frame_number in enumerate(segment):
            frame_to_detections[frame_number] = ball_detections[seg_idx][frame_pos]

    for i, frame in enumerate(video_frames):
        # Check if current frame has ball detections
        if i in frame_to_detections:
            det = frame_to_detections[i]
            for _, bbox in det.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(
                    frame, 
                    (int(x1), int(y1)), 
                    (int(x2), int(y2)), 
                    (0, 255, 255), 2
                )
        
        # Add frame number text (unchanged)
        cv2.putText(
            frame, f"Frame: {i}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        annotated.append(frame)

    # Rest of the code remains unchanged
    output_video_name = f"resolution_{resolution}_fps_{int(fps)}_{ts}.avi"
    output_video_path = os.path.join("output_videos", output_video_name)
    save_video(annotated, output_video_path, fps=fps)
    print("[7/8] Video saved to output_videos/output_video.avi")
    runtimes["save_video"] = time.time() - t0

    # ─── 8) Save run‐metadata JSON ─────────────────────────────────────────────
    t0 = time.time()
    print("[8/8] Writing run metadata JSON…")
    metadata = {
        "video_length_sec": video_length_sec,
        "fps_used": fps,
        "resolution": resolution,
        "segments": [
            {
                "start_frame": s,
                "end_frame": e,
                "start_time_sec": s / fps,
                "end_time_sec": e / fps,
            }
            for s, e in coarse_segs
        ],
        "runtimes": runtimes,
        "total_runtime": time.time() - t_start_total,
    }
    json_path = os.path.join("output_data", f"{ts}.json")
    with open(json_path, "w") as jf:
        json.dump(metadata, jf, indent=4)
    print(f"[8/8] Metadata JSON saved to {json_path}")


if __name__ == "__main__":
    main()
