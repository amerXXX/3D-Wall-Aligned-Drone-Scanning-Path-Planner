#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import json
import asyncio
import pandas as pd
import re
import logging
import cv2
import numpy as np
import PIL.Image
import torch
from sklearn.linear_model import RANSACRegressor
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLOE
import requests
import time
from scipy.spatial.distance import cdist

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================================
# 1. MODEL LOADING AND SETUP
# =====================================================================================
def get_device() -> torch.device:
    """Determines and returns the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        print("Using device: CUDA")
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        print("Using device: MPS")
        return torch.device("mps")
    print("Using device: CPU")
    return torch.device("cpu")

DEVICE = get_device()
print("Loading Depth-Anything-V2 model...")
DA_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
depth_processor = AutoImageProcessor.from_pretrained(DA_MODEL_ID)
depth_model = AutoModelForDepthEstimation.from_pretrained(DA_MODEL_ID).to(DEVICE).eval()
print("Loading YOLOE segmentation model...")
yolo_model = YOLOE("yoloe-11l-seg.pt").to(DEVICE)
PROMPTS = ["wall", "roof", "floor"]
yolo_model.set_classes(PROMPTS, yolo_model.get_text_pe(PROMPTS))
WALL_CLASS_ID = PROMPTS.index("wall")
print(f"Models loaded successfully. 'wall' is class ID {WALL_CLASS_ID}.")

# =====================================================================================
# 2. HELPER FUNCTIONS
# =====================================================================================
def infer_depth(pil_img):
    """Infers depth from a PIL image using Depth-Anything-V2 model."""
    inp = depth_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = depth_model(**inp).predicted_depth
    depth_map = torch.nn.functional.interpolate(
        out.unsqueeze(1), size=pil_img.size[::-1], mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    return depth_map

def clean_mask(mask, kernel_size=7, min_area=1000):
    """Cleans a binary mask using morphological operations and connected components."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    clean = np.zeros_like(opened)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean

def get_wall_mask(image_bgr, model):
    results = model(image_bgr, conf=0.05, iou=0.1)[0]
    masks, cls_ids = results.masks.data, results.boxes.cls.cpu().numpy().astype(int)
    wall_indices = np.where(cls_ids == WALL_CLASS_ID)[0]
    if wall_indices.size == 0:
        print("Warning: No 'wall' detected by the segmentation model.")
        return None
    first_wall_idx = wall_indices[0]
    prob_mask = masks[first_wall_idx]
    raw_mask_255 = (prob_mask > 0.5).cpu().numpy().astype(np.uint8) * 255
    h, w = image_bgr.shape[:2]
    mask_full_size = cv2.resize(raw_mask_255, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_clean_255 = clean_mask(mask_full_size)
    return (mask_clean_255 / 255).astype(np.uint8)

# =====================================================================================
# 3. 3D WALL PLANE ANALYSIS AND COORDINATE SYSTEM
# =====================================================================================
def compute_wall_plane_robust(depth_map, mask, K):
    """
    Computes wall plane equation and establishes 3D coordinate system aligned with wall.
    Returns plane equation, wall coordinate system, and validation metrics.
    """
    K_inv = np.linalg.inv(K)
    v_coords, u_coords = np.where(mask == 1)
    depths = depth_map[v_coords, u_coords]
    
    # Filter out invalid depths
    valid_indices = (depths > 0.1) & (depths < 10.0)  # Reasonable depth range
    if np.sum(valid_indices) < 100:
        logging.error("Not enough valid wall points for plane fitting")
        return None
    
    u_coords, v_coords, depths = u_coords[valid_indices], v_coords[valid_indices], depths[valid_indices]
    
    # Convert to 3D points
    P = (K_inv @ np.stack([u_coords, v_coords, np.ones_like(u_coords)]) * depths).T
    
    # Robust plane fitting with RANSAC
    try:
        # Use all 3 coordinates for more robust plane fitting
        X_plane = P[:, [0, 1]]  # X, Y coordinates
        z_plane = P[:, 2]       # Z coordinates
        
        ransac = RANSACRegressor(
            min_samples=20, 
            max_trials=200, 
            loss='absolute_error', 
            random_state=42,
            stop_n_inliers=int(len(P) * 0.7)
        )
        ransac.fit(X_plane, z_plane)
        
        # Get plane parameters: z = ax + by + c => ax + by - z + c = 0
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_
        
        # Plane normal vector (normalized)
        normal = np.array([a, b, -1.0])
        normal = normal / np.linalg.norm(normal)
        
        # Get inlier points for further analysis
        inlier_points = P[ransac.inlier_mask_]
        centroid = np.mean(inlier_points, axis=0)
        
        # Ensure normal points toward camera (negative Z direction)
        if np.dot(normal, centroid) > 0:
            normal = -normal
        
        # Plane equation: normal · (p - centroid) = 0
        # Or: normal[0]*x + normal[1]*y + normal[2]*z + d = 0
        d = -np.dot(normal, centroid)
        
        plane_params = {
            'normal': normal,
            'centroid': centroid,
            'd': d,
            'inlier_points': inlier_points,
            'inlier_ratio': np.sum(ransac.inlier_mask_) / len(P)
        }
        
        logging.info(f"Wall plane fitted successfully. Inlier ratio: {plane_params['inlier_ratio']:.3f}")
        return plane_params
        
    except Exception as e:
        logging.error(f"Wall plane fitting failed: {e}")
        return None

def create_wall_coordinate_system(plane_params):
    """
    Creates a 3D coordinate system aligned with the wall surface.
    Returns transformation matrices and wall-aligned axes.
    """
    normal = plane_params['normal']
    centroid = plane_params['centroid']
    
    # Create wall-aligned coordinate system
    # Z-axis: wall normal (pointing toward camera)
    wall_z = -normal  # Point toward camera
    
    # X-axis: horizontal direction in wall plane (project world X onto wall)
    world_x = np.array([1.0, 0.0, 0.0])
    wall_x = world_x - np.dot(world_x, normal) * normal
    wall_x = wall_x / np.linalg.norm(wall_x)
    
    # Y-axis: complete the right-handed system
    wall_y = np.cross(wall_z, wall_x)
    wall_y = wall_y / np.linalg.norm(wall_y)
    
    # Create transformation matrix from world to wall coordinates
    # Columns are the wall coordinate system axes expressed in world coordinates
    wall_to_world = np.column_stack([wall_x, wall_y, wall_z])
    world_to_wall = wall_to_world.T
    
    wall_coord_system = {
        'origin': centroid,
        'x_axis': wall_x,  # Horizontal along wall
        'y_axis': wall_y,  # Vertical along wall  
        'z_axis': wall_z,  # Normal to wall (toward camera)
        'wall_to_world': wall_to_world,
        'world_to_wall': world_to_wall
    }
    
    return wall_coord_system

def project_points_to_wall_plane(points_3d, plane_params):
    """
    Projects 3D points onto the wall plane.
    """
    normal = plane_params['normal']
    centroid = plane_params['centroid']
    
    # Vector from centroid to each point
    vectors = points_3d - centroid
    
    # Project vectors onto plane (remove normal component)
    projections = vectors - np.outer(np.dot(vectors, normal), normal)
    
    # Projected points
    projected_points = centroid + projections
    
    return projected_points

# =====================================================================================
# 4. 3D WALL-ALIGNED SCANNING SYSTEM
# =====================================================================================
def generate_wall_aligned_scan_lines(depth_map, mask, K, plane_params, wall_coord_system, scan_spacing_cm=4.0):
    """
    Generates scan lines aligned with the wall surface in 3D space.
    Returns scan lines with 3D coordinates and image projections.
    """
    
    # Get wall bounds in wall coordinate system 
    v_coords, u_coords = np.where(mask == 1)
    depths = depth_map[v_coords, u_coords]
    valid_indices = (depths > 0.1) & (depths < 10.0)
    
    if np.sum(valid_indices) < 50:
        logging.error("Not enough valid points for scan line generation")
        return []
    
    u_coords, v_coords, depths = u_coords[valid_indices], v_coords[valid_indices], depths[valid_indices]
    
    # Convert to 3D world coordinates
    K_inv = np.linalg.inv(K)
    points_3d_world = (K_inv @ np.stack([u_coords, v_coords, np.ones_like(u_coords)]) * depths).T
    
    # Project points onto wall plane
    points_3d_projected = project_points_to_wall_plane(points_3d_world, plane_params)
    
    # Transform to wall coordinate system
    origin = wall_coord_system['origin']
    world_to_wall = wall_coord_system['world_to_wall']
    
    points_wall_coords = (points_3d_projected - origin) @ world_to_wall.T
    
    # Get wall bounds in wall coordinates
    wall_x_min, wall_x_max = np.min(points_wall_coords[:, 0]), np.max(points_wall_coords[:, 0])
    wall_y_min, wall_y_max = np.min(points_wall_coords[:, 1]), np.max(points_wall_coords[:, 1])
    
    logging.info(f"Wall bounds in wall coordinates: X=[{wall_x_min:.2f}, {wall_x_max:.2f}], Y=[{wall_y_min:.2f}, {wall_y_max:.2f}] meters")
    
    # Generate scan lines (horizontal lines in wall coordinate system)
    scan_spacing_m = scan_spacing_cm / 100.0
    scan_lines_data = []
    
    y_positions = np.arange(wall_y_min, wall_y_max, scan_spacing_m)
    
    for i, y_wall in enumerate(y_positions):
        # Create scan line in wall coordinates (horizontal line at constant Y)
        x_positions = np.linspace(wall_x_min, wall_x_max, int((wall_x_max - wall_x_min) / 0.02) + 1)  # 2cm steps
        
        scan_line_points_wall = np.column_stack([
            x_positions,
            np.full_like(x_positions, y_wall),
            np.zeros_like(x_positions)  # On the wall plane (z=0 in wall coords)
        ])
        
        # Transform back to world coordinates
        scan_line_points_world = origin + (scan_line_points_wall @ wall_coord_system['wall_to_world'].T)
        
        # Project to image coordinates
        scan_line_points_image = []
        scan_line_segments = []
        
        for point_3d in scan_line_points_world:
            # Project 3D point to image
            point_homogeneous = K @ point_3d
            if point_homogeneous[2] > 0:  # Valid projection
                u = int(point_homogeneous[0] / point_homogeneous[2])
                v = int(point_homogeneous[1] / point_homogeneous[2])
                
                # Check if point is within image bounds
                if 0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]:
                    scan_line_points_image.append((u, v))
        
        if len(scan_line_points_image) < 2:
            continue
            
        # Analyze segments along the scan line
        segments = analyze_scan_line_segments(scan_line_points_image, mask)
        
        if segments:
            scan_line_data = {
                'line_id': i,
                'y_wall_coord': y_wall,
                'points_3d_world': scan_line_points_world,
                'points_2d_image': scan_line_points_image,
                'segments': segments
            }
            scan_lines_data.append(scan_line_data)
    
    logging.info(f"Generated {len(scan_lines_data)} wall-aligned scan lines")
    return scan_lines_data

def analyze_scan_line_segments(image_points, mask):
    """
    Analyzes segments along a scan line to identify wall vs obstacle regions.
    """
    if len(image_points) < 2:
        return []
    
    segments = []
    current_type = None
    segment_start_idx = 0
    
    for i, (u, v) in enumerate(image_points):
        # Determine segment type based on mask
        pixel_type = 'wall' if mask[v, u] == 1 else 'obstacle'
        
        if current_type is None:
            current_type = pixel_type
            segment_start_idx = i
        elif pixel_type != current_type:
            # End current segment
            if i > segment_start_idx:
                segment_length_cm = (i - segment_start_idx) * 2.0  # Approximate 2cm per point
                segments.append({
                    'type': current_type,
                    'start_idx': segment_start_idx,
                    'end_idx': i - 1,
                    'length_cm': segment_length_cm,
                    'start_point': image_points[segment_start_idx],
                    'end_point': image_points[i - 1]
                })
            
            # Start new segment
            current_type = pixel_type
            segment_start_idx = i
    
    # Add final segment
    if current_type is not None and len(image_points) > segment_start_idx:
        segment_length_cm = (len(image_points) - segment_start_idx) * 2.0
        segments.append({
            'type': current_type,
            'start_idx': segment_start_idx,
            'end_idx': len(image_points) - 1,
            'length_cm': segment_length_cm,
            'start_point': image_points[segment_start_idx],
            'end_point': image_points[-1]
        })
    
    return segments

def calculate_3d_scan_distances(wall_coord_system):
    """
    Calculates real-world distances for drone movements based on wall coordinate system.
    """
    # Standard scanning parameters in real-world units
    scan_step_cm = 2.0  # 2cm steps along wall surface
    line_spacing_cm = 4.0  # 4cm between scan lines
    safe_distance_cm = 50.0  # 50cm from wall
    
    distances = {
        'scan_step_cm': scan_step_cm,
        'line_spacing_cm': line_spacing_cm,
        'safe_distance_cm': safe_distance_cm,
        'wall_coordinate_system': wall_coord_system
    }
    
    return distances

# =====================================================================================
# 5. ENHANCED INITIAL POSITIONING SYSTEM
# =====================================================================================
def calculate_initial_drone_positioning(wall_coord_system, scan_lines_data, K):
    """
    Calculates initial drone positioning to align with wall coordinate system.
    """
    if not scan_lines_data:
        logging.error("No scan lines available for initial positioning")
        return []
    
    # Find the starting scan line (topmost)
    start_line = scan_lines_data[0]
    
    # Find leftmost wall segment in the starting line
    wall_segments = [seg for seg in start_line['segments'] if seg['type'] == 'wall']
    if not wall_segments:
        logging.error("No wall segments found in starting scan line")
        return []
    
    start_segment = wall_segments[0]
    start_point_image = start_segment['start_point']
    
    # Get 3D coordinates of start point
    start_point_3d_world = None
    for i, (u, v) in enumerate(start_line['points_2d_image']):
        if (u, v) == start_point_image:
            start_point_3d_world = start_line['points_3d_world'][i]
            break
    
    if start_point_3d_world is None:
        logging.error("Could not find 3D coordinates for start point")
        return []
    
    # Calculate drone positioning relative to wall coordinate system
    origin = wall_coord_system['origin']
    wall_normal = wall_coord_system['z_axis']
    
    # Target drone position: 50cm away from wall, aligned with start point
    safe_distance_m = 0.5  # 50cm
    target_drone_position = start_point_3d_world + wall_normal * safe_distance_m
    
    # Calculate movement from image center (assumed initial drone position)
    H, W = 720, 1280  # Image dimensions
    image_center_3d = np.array([0, 0, 2.0])  # Assume 2m initial distance
    
    movement_vector = target_drone_position - image_center_3d
    
    # Convert to drone commands
    initial_commands = []
    
    # Rotation to align with wall
    wall_yaw_rad = math.atan2(wall_normal[0], wall_normal[2])
    wall_yaw_deg = math.degrees(wall_yaw_rad)
    
    if abs(wall_yaw_deg) > 2:  # Threshold for rotation
        initial_commands.append(f"tello.rotate_tello_by_angle {int(round(wall_yaw_deg))}")
    
    # Translation movements (convert from meters to cm)
    movement_x_cm = movement_vector[0] * 100
    movement_y_cm = movement_vector[1] * 100
    movement_z_cm = movement_vector[2] * 100
    
    # Forward/backward movement (Z-axis)
    if abs(movement_z_cm) > 5:
        if movement_z_cm > 0:
            initial_commands.append(f"tello.move_forward {int(round(abs(movement_z_cm)))}")
        else:
            initial_commands.append(f"tello.move_backward {int(round(abs(movement_z_cm)))}")
    
    # Left/right movement (X-axis)
    if abs(movement_x_cm) > 5:
        if movement_x_cm > 0:
            initial_commands.append(f"tello.move_right {int(round(abs(movement_x_cm)))}")
        else:
            initial_commands.append(f"tello.move_left {int(round(abs(movement_x_cm)))}")
    
    # Up/down movement (Y-axis, note: image Y is inverted)
    if abs(movement_y_cm) > 5:
        if movement_y_cm < 0:  # Negative Y means move up
            initial_commands.append(f"tello.move_up {int(round(abs(movement_y_cm)))}")
        else:
            initial_commands.append(f"tello.move_down {int(round(abs(movement_y_cm)))}")
    
    logging.info(f"Calculated {len(initial_commands)} initial positioning commands")
    return initial_commands

# =====================================================================================
# 6. ENHANCED LLM PATH PLANNING WITH 3D AWARENESS
# =====================================================================================
async def llm_path_planning_3d(scan_lines_data, wall_coord_system):
    """
    Enhanced LLM path planning using 3D wall-aligned scan lines.
    """
    all_tello_commands = []
    
    # Initialize drone state with 3D awareness
    drone_state = {
        'direction': 'left_to_right',
        'current_location': 'start_of_line',
        'is_first_line': True,
        'is_last_line': False,
        'upcoming_obstacle_on_next_line': False,
        'wall_coordinate_system': {
            'x_axis_world': wall_coord_system['x_axis'].tolist(),
            'y_axis_world': wall_coord_system['y_axis'].tolist(),
            'z_axis_world': wall_coord_system['z_axis'].tolist()
        }
    }
    
    num_lines = len(scan_lines_data)
    
    for i, line_data in enumerate(scan_lines_data):
        # Update state
        drone_state['is_first_line'] = (i == 0)
        drone_state['is_last_line'] = (i == num_lines - 1)
        
        # Check upcoming obstacles
        if not drone_state['is_last_line']:
            next_line_data = scan_lines_data[i + 1]
            first_segment_next_line = next_line_data['segments'][0] if next_line_data['segments'] else None
            drone_state['upcoming_obstacle_on_next_line'] = (
                first_segment_next_line and first_segment_next_line['type'] == 'obstacle'
            )
        else:
            drone_state['upcoming_obstacle_on_next_line'] = False
        
        # Prepare simplified line data for LLM (remove large arrays)
        simplified_line_data = {
            'line_id': line_data['line_id'],
            'y_wall_coord': line_data['y_wall_coord'],
            'segments': line_data['segments']
        }
        
        line_str = json.dumps(simplified_line_data, indent=2)
        state_str = json.dumps(drone_state, indent=2)
        
        # Enhanced prompt with 3D awareness
        prompt = f"""
You are an expert Tello drone path planner working with 3D wall-aligned scanning.

## Current Situation:
The drone is scanning a wall using a 3D coordinate system aligned with the wall surface. 
Scan lines are generated parallel to the wall surface, not the image plane.
All movements are calculated in real-world 3D coordinates.

## Drone State:
{state_str}

## Current 3D Wall-Aligned Scan Line:
{line_str}

## 3D Scanning Rules:
- The scan lines are aligned with the actual wall surface in 3D space
- Scan 'wall' segments in 20cm steps (real-world distance along wall surface)
- Avoid 'obstacle' segments completely
- Move between scan lines using 4cm vertical steps (along wall surface)
- Maintain 50cm distance from wall surface at all times
- Use zigzag pattern (alternate direction each line)

## Movement Commands:
- `tello.move_left <distance_cm> wall` - Move left along wall surface
- `tello.move_right <distance_cm> wall` - Move right along wall surface  
- `tello.move_down <distance_cm> wall` - Move down along wall surface (between lines)
- `tello.move_backward <distance_cm> obstacle` - Move away from wall to avoid obstacle
- `tello.move_forward <distance_cm> obstacle` - Move back toward wall after avoiding obstacle

## Obstacle Avoidance (3D-aware):
When encountering obstacles:
1. `tello.move_backward 50 obstacle` (move away from wall)
2. `tello.move_left/right <obstacle_width_cm> obstacle` (go around obstacle)
3. `tello.move_forward 50 obstacle` (return to wall) - ONLY if next segment is wall

## Output Format:
- Output ONLY Tello movement commands 
- One command per line
- Format: `tello.<action> <value_cm> <type>`
- NO explanations, comments, or markdown

Generate the movement commands for this scan line:
"""
        
        # Call LLM API
        apiKey = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your actual API key
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192}
        }
        
        try:
            logging.info(f"Calling Gemini API for 3D scan line {i+1}/{num_lines}...")
            response = await asyncio.to_thread(requests.post,
                apiUrl,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            result = response.json()
            
            gemini_text = ""
            if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
                gemini_text = result['candidates'][0]['content']['parts'][0]['text']
            else:
                logging.error(f"Gemini API response structure unexpected for line {i}: {result}")
                gemini_text = f"Error: Could not generate commands for line {i}"
            
            # Parse and add commands
            commands_for_line = gemini_text.strip().split('\n')
            all_tello_commands.extend([cmd for cmd in commands_for_line if cmd.strip()])
            
            # Update drone state
            drone_state['direction'] = 'right_to_left' if drone_state['direction'] == 'left_to_right' else 'left_to_right'
            drone_state['current_location'] = 'start_of_line'
            
        except Exception as e:
            logging.error(f"Error calling Gemini API for line {i}: {e}")
            all_tello_commands.append(f"Error: Could not generate commands for line {i} (API error: {e})")
            break
        
        # Rate limiting
        logging.info(f"API call successful for 3D line {i+1}. Waiting 10 seconds...")
        await asyncio.sleep(10)
    
    return "\n".join(all_tello_commands)

# =====================================================================================
# 7. MAIN EXECUTION WITH 3D WALL ALIGNMENT
# =====================================================================================
async def main(args):
    # Camera intrinsics
    W, H = 1280, 720
    fx, fy = 960.0, 962.0
    cx, cy = 528.47, 365.47
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    output_dir = args.output_path if args.output_path else Path("./")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Processing image with 3D wall-aligned scanning: {args.image_path}")
    
    # Load and process image
    image_bgr = cv2.imread(str(args.image_path))
    if image_bgr is None:
        logging.error(f"Could not load image at {args.image_path}")
        raise FileNotFoundError(f"Could not load image at {args.image_path}")
    
    h_orig, w_orig = image_bgr.shape[:2]
    if (w_orig, h_orig) != (W, H):
        logging.warning(f"Resizing image from {w_orig}x{h_orig} to {W}x{H}")
        image_bgr = cv2.resize(image_bgr, (W, H))
    
    # Get wall mask
    logging.info("Segmenting wall...")
    wall_mask = get_wall_mask(image_bgr, yolo_model)
    if wall_mask is None:
        logging.error("No wall detected")
        return
    
    # Get depth map
    logging.info("Estimating depth...")
    image_pil = PIL.Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    depth_map = infer_depth(image_pil)
    
    # Compute 3D wall plane
    logging.info("Computing 3D wall plane...")
    plane_params = compute_wall_plane_robust(depth_map, wall_mask, K)
    if plane_params is None:
        logging.error("Failed to compute wall plane")
        return
    
    # Create wall coordinate system
    logging.info("Creating wall-aligned coordinate system...")
    wall_coord_system = create_wall_coordinate_system(plane_params)
    
    # Generate wall-aligned scan lines
    logging.info("Generating 3D wall-aligned scan lines...")
    scan_lines_data = generate_wall_aligned_scan_lines(
        depth_map, wall_mask, K, plane_params, wall_coord_system, 
        scan_spacing_cm=args.scan_step_pixels / 10.0  # Convert pixel spacing to approximate cm
    )
    
    if not scan_lines_data:
        logging.error("No valid scan lines generated")
        return
    
    # Calculate initial drone positioning
    logging.info("Calculating initial drone positioning...")
    initial_commands = calculate_initial_drone_positioning(wall_coord_system, scan_lines_data, K)
    
    # Generate 3D-aware path planning
    logging.info("Generating 3D wall-aligned path plan...")
    tello_commands_str = await llm_path_planning_3d(scan_lines_data, wall_coord_system)
    
    # Process commands for Excel export
    final_action_value_pairs = []
    
    # Add initial commands
    for cmd in initial_commands:
        parts = cmd.split()
        if len(parts) >= 2 and parts[0].startswith("tello."):
            action = parts[0].replace("tello.", "")
            try:
                value = int(parts[1])
                final_action_value_pairs.append((action, value, "initial"))
            except ValueError:
                logging.warning(f"Could not parse initial command: {cmd}")
    
    # Add LLM-generated commands
    if "Error:" not in tello_commands_str:
        command_lines = tello_commands_str.strip().split('\n')
        command_regex = re.compile(r"tello\.(\w+)\s+(\d+)\s+(wall|obstacle|initial)")
        
        for cmd in command_lines:
            cmd = cmd.strip()
            if not cmd:
                continue
            match = command_regex.match(cmd)
            if match:
                action, value_str, cmd_type = match.groups()
                try:
                    value = int(value_str)
                    final_action_value_pairs.append((action, value, cmd_type))
                except ValueError:
                    logging.warning(f"Could not convert value in command: {cmd}")
            else:
                logging.warning(f"Skipping invalid command format: {cmd}")
    
    # Save results to Excel
    if final_action_value_pairs:
        try:
            df_final = pd.DataFrame(final_action_value_pairs, columns=["action", "value", "type"])
            excel_output_file = output_dir / f"{args.image_path.stem}_3d_tello_commands.xlsx"
            df_final.to_excel(excel_output_file, index=False)
            logging.info(f"Saved 3D-aligned Tello commands to: {excel_output_file}")
            
            # Save detailed scan line data
            scan_data_file = output_dir / f"{args.image_path.stem}_3d_scan_data.json"
            scan_data_export = []
            for line in scan_lines_data:
                line_export = {
                    'line_id': line['line_id'],
                    'y_wall_coord': float(line['y_wall_coord']),
                    'segments': line['segments']
                }
                scan_data_export.append(line_export)
            
            with open(scan_data_file, 'w') as f:
                json.dump(scan_data_export, f, indent=2)
            logging.info(f"Saved 3D scan data to: {scan_data_file}")
            
        except Exception as e:
            logging.error(f"Error saving results: {e}")
    
    # Create comprehensive visualization
    logging.info("Creating 3D wall-aligned visualization...")
    create_3d_visualization(image_bgr, wall_mask, depth_map, plane_params, 
                           wall_coord_system, scan_lines_data, K, output_dir, args.image_path.stem)
    
    # Save wall coordinate system info
    coord_system_info = {
        'wall_origin': wall_coord_system['origin'].tolist(),
        'wall_x_axis': wall_coord_system['x_axis'].tolist(),
        'wall_y_axis': wall_coord_system['y_axis'].tolist(), 
        'wall_z_axis': wall_coord_system['z_axis'].tolist(),
        'wall_normal': plane_params['normal'].tolist(),
        'wall_centroid': plane_params['centroid'].tolist(),
        'inlier_ratio': plane_params['inlier_ratio']
    }
    
    coord_file = output_dir / f"{args.image_path.stem}_wall_coordinate_system.json"
    with open(coord_file, 'w') as f:
        json.dump(coord_system_info, f, indent=2)
    logging.info(f"Saved wall coordinate system info to: {coord_file}")
    
    logging.info("3D wall-aligned processing completed successfully!")

def create_3d_visualization(image_bgr, wall_mask, depth_map, plane_params, wall_coord_system, 
                           scan_lines_data, K, output_dir, image_stem):
    """
    Creates comprehensive visualization of 3D wall-aligned scanning system.
    """
    
    # Create main visualization
    vis_image = image_bgr.copy()
    
    # Draw wall mask overlay
    wall_overlay = np.zeros_like(vis_image)
    wall_overlay[wall_mask == 1] = [0, 255, 0]  # Green for wall
    vis_image = cv2.addWeighted(vis_image, 0.7, wall_overlay, 0.3, 0)
    
    # Draw 3D scan lines projected to image
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, line_data in enumerate(scan_lines_data):
        color = colors[i % len(colors)]
        points_2d = line_data['points_2d_image']
        
        # Draw scan line
        for j in range(len(points_2d) - 1):
            pt1 = points_2d[j]
            pt2 = points_2d[j + 1]
            cv2.line(vis_image, pt1, pt2, color, 2)
        
        # Draw segments
        for segment in line_data['segments']:
            start_pt = segment['start_point']
            end_pt = segment['end_point']
            seg_color = (0, 255, 0) if segment['type'] == 'wall' else (0, 0, 255)
            cv2.line(vis_image, start_pt, end_pt, seg_color, 4)
            
            # Add segment labels
            mid_x = (start_pt[0] + end_pt[0]) // 2
            mid_y = (start_pt[1] + end_pt[1]) // 2
            cv2.putText(vis_image, f"{segment['type'][:1]}{segment['length_cm']:.0f}cm", 
                       (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, seg_color, 1)
    
    # Draw coordinate system origin
    origin_3d = wall_coord_system['origin']
    origin_2d = project_3d_to_image(origin_3d, K)
    if origin_2d is not None:
        cv2.circle(vis_image, origin_2d, 8, (255, 255, 255), -1)
        cv2.putText(vis_image, "Wall Origin", (origin_2d[0] + 10, origin_2d[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw coordinate axes
    axis_length = 0.2  # 20cm axes
    x_axis_end = origin_3d + wall_coord_system['x_axis'] * axis_length
    y_axis_end = origin_3d + wall_coord_system['y_axis'] * axis_length
    z_axis_end = origin_3d + wall_coord_system['z_axis'] * axis_length
    
    x_axis_2d = project_3d_to_image(x_axis_end, K)
    y_axis_2d = project_3d_to_image(y_axis_end, K)
    z_axis_2d = project_3d_to_image(z_axis_end, K)
    
    if origin_2d and x_axis_2d:
        cv2.arrowedLine(vis_image, origin_2d, x_axis_2d, (0, 0, 255), 3)  # Red X
        cv2.putText(vis_image, "X", x_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    if origin_2d and y_axis_2d:
        cv2.arrowedLine(vis_image, origin_2d, y_axis_2d, (0, 255, 0), 3)  # Green Y
        cv2.putText(vis_image, "Y", y_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if origin_2d and z_axis_2d:
        cv2.arrowedLine(vis_image, origin_2d, z_axis_2d, (255, 0, 0), 3)  # Blue Z
        cv2.putText(vis_image, "Z", z_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Add information text
    info_text = [
        f"3D Wall-Aligned Scanning",
        f"Scan Lines: {len(scan_lines_data)}",
        f"Wall Normal: [{plane_params['normal'][0]:.2f}, {plane_params['normal'][1]:.2f}, {plane_params['normal'][2]:.2f}]",
        f"Inlier Ratio: {plane_params['inlier_ratio']:.2f}",
        f"Wall Origin: [{origin_3d[0]:.2f}, {origin_3d[1]:.2f}, {origin_3d[2]:.2f}]m"
    ]
    
    y_offset = 30
    for i, text in enumerate(info_text):
        cv2.putText(vis_image, text, (10, y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save visualization
    vis_output_file = output_dir / f"{image_stem}_3d_wall_aligned_visualization.png"
    cv2.imwrite(str(vis_output_file), vis_image)
    logging.info(f"Saved 3D visualization to: {vis_output_file}")
    
    # Create depth visualization
    depth_vis = cv2.applyColorMap((depth_map / np.max(depth_map) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_output_file = output_dir / f"{image_stem}_depth_visualization.png"
    cv2.imwrite(str(depth_output_file), depth_vis)
    logging.info(f"Saved depth visualization to: {depth_output_file}")

def project_3d_to_image(point_3d, K):
    """
    Projects a 3D point to image coordinates.
    """
    if point_3d[2] <= 0:
        return None
    
    point_homogeneous = K @ point_3d
    u = int(point_homogeneous[0] / point_homogeneous[2])
    v = int(point_homogeneous[1] / point_homogeneous[2])
    
    return (u, v)

# =====================================================================================
# 8. RUNNER
# =====================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Wall-Aligned Pose Estimation and Tello Path Planning.")
    parser.add_argument("--image_path", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--output_path", type=Path, help="Directory to save outputs.")
    parser.add_argument("--scan_step_pixels", type=int, default=40, help="Approximate spacing for scan lines (converted to real-world units).")
    args = parser.parse_args()
    
    asyncio.run(main(args))