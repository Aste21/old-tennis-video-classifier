import os
import sys

# ─── Ensure project root is on the module search path ────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.video_utils import read_video
import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np


class CourtLineDetector:
    def __init__(self, model_path):
        # 1) Start from an ImageNet-pretrained ResNet50:
        self.model = models.resnet50(pretrained=True)
        # 2) Replace final FC with a regression head for 14 keypoints (x,y):
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        # 3) Standard ImageNet transforms:
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image):
        """
        Returns a flat numpy array of length 28:
        [x0, y0,  x1, y1,  …,  x13, y13]
        all in pixel coords of the ORIGINAL image.
        """
        # BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
        return keypoints

    def is_court_frame(
        self, image, min_area_ratio=0.25, aspect_range=(1.0, 3.0)
    ):
        """
        Returns True if the predicted court keypoints cover a plausible tennis-court area.
        """
        kp = self.predict(image)
        xs, ys = kp[::2], kp[1::2]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        width, height = x_max - x_min, y_max - y_min
        area_ratio = (width * height) / (image.shape[1] * image.shape[0])
        aspect = width / (height + 1e-6)
        return (area_ratio > min_area_ratio) and (
            aspect_range[0] < aspect < aspect_range[1]
        )

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i + 1])
            cv2.putText(
                image,
                str(i // 2),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output = []
        for frame in video_frames:
            output.append(self.draw_keypoints(frame.copy(), keypoints))
        return output


def _format_hms(seconds: float) -> str:
    """Helper: converts seconds to H:MM:SS string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def test():
    """
    Debug test:
    - Loads input_videos/input1.mp4
    - Samples 1 frame/sec
    - For the *first* sampled frame, prints raw keypoints and saves a debug image
    - For each sample, prints width, height, area_ratio, aspect, and uses
      is_court_frame() to decide court vs. no-court
    - Collapses and prints detected segments
    """
    video_path = os.path.join(PROJECT_ROOT, "input_videos", "input2.mp4")
    print(f"Testing on video: {video_path}")

    frames = read_video(video_path)
    if not frames:
        print(f"Could not read frames from {video_path}")
        return

    model_path = os.path.join(PROJECT_ROOT, "models", "keypoints_model.pth")
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        return
    cld = CourtLineDetector(model_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    sample_rate_sec = 1.0
    step = max(1, int(fps * sample_rate_sec))

    timeline = []
    debug_done = False
    for idx, frame in enumerate(frames):
        if idx % step != 0:
            continue
        t = idx / fps

        # 1) get raw keypoints for debug
        kp = cld.predict(frame)
        xs, ys = kp[::2], kp[1::2]
        w_box = float(xs.max() - xs.min())
        h_box = float(ys.max() - ys.min())
        area_ratio = (w_box * h_box) / (frame.shape[1] * frame.shape[0])
        aspect = w_box / (h_box + 1e-6)

        # 2) Delegate to your static method under test
        flag = cld.is_court_frame(frame)

        # 3) print debug info
        print(f"[{_format_hms(t)}] w={w_box:.1f}, h={h_box:.1f}, "
              f"area_ratio={area_ratio:.3f}, aspect={aspect:.3f}, court={flag}")

        # 4) On the *first* sampled frame, dump keypoints and save an image
        if not debug_done:
            print("\n First sample keypoints (x,y for i=0..13):")
            for i in range(14):
                print(f"  kp[{i}]: ({xs[i]:.1f}, {ys[i]:.1f})")
            vis = cld.draw_keypoints(frame.copy(), kp)
            out_path = os.path.join(PROJECT_ROOT, "output_videos", "debug_keypoints.png")
            cv2.imwrite(out_path, vis)
            print(f"Debug image with keypoints saved to: {out_path}\n")
            debug_done = True

        timeline.append((t, flag))

    # collapse into segments
    segments = []
    seg_start = None
    for t, present in timeline:
        if present and seg_start is None:
            seg_start = t
        elif not present and seg_start is not None:
            segments.append((seg_start, t))
            seg_start = None
    if seg_start is not None:
        segments.append((seg_start, timeline[-1][0]))

    # report
    if not segments:
        print("No tennis court detected in this video.")
    else:
        print("\nTennis-court segments detected:")
        for start, end in segments:
            print(f"   {_format_hms(start)} → {_format_hms(end)}")

if __name__ == "__main__":
    test()