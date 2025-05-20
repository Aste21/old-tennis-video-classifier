import matplotlib.pyplot as plt

def plot_ball_trajectory(segment_detections, output_path, segment_number, court_dimensions=None):
    """Plot ball trajectory for a single segment and save to file."""
    plt.figure(figsize=(12, 8))
    
    x_coords = []
    y_coords = []
    
    for frame in segment_detections:
        if 1 in frame:  # Check if ball was detected in this frame
            bbox = frame[1]
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            x_coords.append(x_center)
            y_coords.append(y_center)
    
    if len(x_coords) == 0:  # No detections in this segment
        plt.text(0.5, 0.5, 'No Ball Detections', ha='center', va='center')
    else:
        plt.plot(x_coords, y_coords, 'b-o', markersize=4, linewidth=1, alpha=0.7)
        plt.scatter(x_coords[0], y_coords[0], c='green', label='Start', zorder=5)
        plt.scatter(x_coords[-1], y_coords[-1], c='red', label='End', zorder=5)
        
        if court_dimensions:
            plt.xlim(0, court_dimensions[0])
            plt.ylim(court_dimensions[1], 0)  # Invert y-axis for image coordinates
        plt.legend()
    
    plt.title(f"Ball Trajectory - Segment {segment_number}")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()