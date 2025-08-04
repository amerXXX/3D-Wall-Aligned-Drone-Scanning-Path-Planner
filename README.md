# 3D Wall Scanner

This project implements a 3D wall scanning system using computer vision and depth estimation. It uses the Depth-Anything-V2 model for depth estimation and YOLOE for wall segmentation.


### Scan Data Processing

The core logic for preparing the scan data for the LLM planner follows a three-step process:

1.  **Vertical Sampling**  
    The script steps every 10px (~20cm) vertically across the detected wall mask to create horizontal scan lines.

2.  **Segment Extraction**  
    Each scan line is then analyzed to find contiguous segments. These are classified by their type (wall or obstacle) and their start/end coordinates, structured as: `[type, x_start, x_end]`.

3.  **Data Serialization**  
    This structured list of segments is converted into a JSON format, which serves as the direct input for the LLM path planner.

**Note:** The first scan line is always assumed to start on a wall segment. Subsequent lines, however, may begin with an obstacle, and the drone must navigate accordingly.

<img width="1150" height="646" alt="image" src="https://github.com/user-attachments/assets/98e926a0-1d15-4716-9c5e-867e43542dc1" />


### Depth Model Characteristics

It's important to understand the characteristics of the monocular depth estimation model (`Depth-Anything-V2`) used in this project:

*   **Precision**: Per-pixel depth data enables highly precise calculations for 3D plane fitting.
*   **Relative Scale**: The model produces relative depth, not a true metric scale. The script cleverly leverages this for plane orientation, but absolute distance relies on the model's training for metric-like output.
*   **Lighting Dependency**: Accuracy can vary depending on the lighting conditions in the input image.
*   **Versatility**: The chosen model supports both indoor and outdoor environments.
<img width="679" height="831" alt="image" src="https://github.com/user-attachments/assets/0abfb9fd-5e69-461b-96b4-b5bf8e3f9c5e" />


## Triangle-Based Alignment Strategy

Before scanning, the drone must perform an initial alignment to position itself correctly at the start of the first scan line. This is achieved in a single "one-shot" maneuver without the need for external markers.

<img width="591" height="388" alt="image" src="https://github.com/user-attachments/assets/879894a4-8af4-4760-814f-81a28332e9f0" />

<img width="804" height="541" alt="image" src="https://github.com/user-attachments/assets/dd96af33-1d40-4d52-8b15-c395ff57316d" />


### Triangulation Process

1.  **Form a Right-Triangle**: A virtual right-triangle is created between three key points: the target **Start Pixel (S)** on the wall, the **Image Center (C)** (representing the drone's current orientation), and a calculated **Drop-Down Point (T)** on the same vertical plane as the start pixel.

2.  **Convert to 3D**: The pixel coordinates and depth value `(u, v, depth)` for the start point are converted into real-world 3D coordinates `(x, y, z)` using the camera's intrinsic parameters.

3.  **Compute Positional Deltas**: The script calculates the required movement along each axis (`Δx`, `Δy`, `Δz`) to move the drone from its current position to the target start position.

### Command Sequence

The calculated deltas are translated directly into a sequence of drone commands, typically following this pattern:
> move **forward/backward** (Δz) → move **left/right** (Δx) → move **up/down** (Δy) → **rotate** (θ)

-   **Results**: This process achieves a one-shot alignment with the wall, preparing the drone to immediately begin scanning.
-   **Advantage**: It completely eliminates the need for external fiducial markers (like QR codes or ArUco markers) or complex CNN-based object detectors for initial positioning.
## Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required model file:
- Download `yoloe-11l-seg.pt` and place it in the root directory
- The Depth-Anything-V2 model will be downloaded automatically on first run

## Usage

Run the script with an input image:
```bash
python LLMX.py --image_path path/to/your/image.jpg --output_path path/to/output
```

Arguments:
- `--image_path`: Path to the input image (required)
- `--output_path`: Directory to save outputs (optional)
- `--scan_step_pixels`: Approximate spacing for scan lines in pixels (default: 40)

## Output

The script generates:
- Wall segmentation visualization
- Depth map visualization
- 3D wall-aligned scan lines
- Coordinate system information
- Initial positioning commands

## Dependencies

See `requirements.txt` for the full list of Python dependencies.

## Models Used

- Depth-Anything-V2 for depth estimation
- YOLOE for wall segmentation

## License

[Your chosen license]


