from ultralytics import YOLO
import cv2
import pickle
import pandas as pd


class BallTracker:
    def __init__(self, model_path, max_ball_distance=1000000):
        self.model = YOLO(model_path)
        self.max_ball_distance = max_ball_distance

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        df_ball_positions["ball_hit"] = 0

        df_ball_positions["mid_y"] = (
            df_ball_positions["y1"] + df_ball_positions["y2"]
        ) / 2
        df_ball_positions["mid_y_rolling_mean"] = (
            df_ball_positions["mid_y"]
            .rolling(window=5, min_periods=1, center=False)
            .mean()
        )
        df_ball_positions["delta_y"] = df_ball_positions["mid_y_rolling_mean"].diff()
        minimum_change_frames_for_hit = 25
        for i in range(
            1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)
        ):
            negative_position_change = (
                df_ball_positions["delta_y"].iloc[i] > 0
                and df_ball_positions["delta_y"].iloc[i + 1] < 0
            )
            positive_position_change = (
                df_ball_positions["delta_y"].iloc[i] < 0
                and df_ball_positions["delta_y"].iloc[i + 1] > 0
            )

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(
                    i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1
                ):
                    negative_position_change_following_frame = (
                        df_ball_positions["delta_y"].iloc[i] > 0
                        and df_ball_positions["delta_y"].iloc[change_frame] < 0
                    )
                    positive_position_change_following_frame = (
                        df_ball_positions["delta_y"].iloc[i] < 0
                        and df_ball_positions["delta_y"].iloc[change_frame] > 0
                    )

                    if (
                        negative_position_change
                        and negative_position_change_following_frame
                    ):
                        change_count += 1
                    elif (
                        positive_position_change
                        and positive_position_change_following_frame
                    ):
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions["ball_hit"].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[
            df_ball_positions["ball_hit"] == 1
        ].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self, frames):
        ball_detections = []
        last_valid_position = None
        for frame_number, frame in enumerate(frames):
            print(f"Detecting frame number {frame_number} out of {len(frames)} frames.")
            player_dict, last_valid_position = self.detect_frame(
                frame, last_valid_position
            )
            ball_detections.append(player_dict)

        return ball_detections

    def detect_frame(self, frame, last_valid_position=None):
        results = self.model.predict(frame, conf=0.15)[0]
        ball_dict = {}
        is_changed = False
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            if last_valid_position is not None:
                print(
                    f"Distance = {self.calc_ball_distance(result, last_valid_position)}"
                )
                if (
                    self.calc_ball_distance(result, last_valid_position)
                    <= self.max_ball_distance
                ):
                    ball_dict[1] = result
                    is_changed = True
            else:
                ball_dict[1] = result
                is_changed = True
            if result is not None and is_changed:
                last_valid_position = result
                break

        return ball_dict, last_valid_position

    def calc_ball_distance(self, dist1, dist2):
        # Calculate center coordinates of the first bounding box
        x1_center = (dist1[0] + dist1[2]) / 2
        y1_center = (dist1[1] + dist1[3]) / 2

        # Calculate center coordinates of the second bounding box
        x2_center = (dist2[0] + dist2[2]) / 2
        y2_center = (dist2[1] + dist2[3]) / 2

        # Compute Euclidean distance between the centers
        distance = ((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2) ** 0.5
        return distance

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Ball ID: {track_id}",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2
                )
            output_video_frames.append(frame)

        return output_video_frames
