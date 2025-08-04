# 3D Wall Scanner

This project implements a 3D wall scanning system using computer vision and depth estimation. It uses the Depth-Anything-V2 model for depth estimation and YOLOE for wall segmentation.

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
