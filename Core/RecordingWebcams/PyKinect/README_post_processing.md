# Post-Processing Tool for Model Training

This tool post-processes video recordings based on gesture labels from the `auto_labels_*.csv` file, creating training-ready video segments optimized for machine learning models.

## 🎯 **Key Features for Model Training**

- **Fixed 15-Second Segments**: Each gesture starts a 15-second video segment
- **Reading Time Exclusion**: Automatically cuts off the first 5 seconds (reading time, not performance time)
- **Training-Optimized Format**: Standardizes videos to 30 FPS for consistent training data
- **Quality Filtering**: Ensures minimum segment duration (3 seconds) for meaningful training samples
- **Standardized Naming**: Creates training-friendly filenames with participant and camera information
- **Multi-Camera Support**: Processes all available cameras (webcams + Azure Kinect color/depth/IR)

## 📁 **Output Structure**

```
dataset/
└── post-processed/
    └── {participant_id}/
        ├── camera_0/
        │   ├── p0_cam0_seg000_Touch_Dashboard.mp4
        │   ├── p0_cam0_seg001_Bouncing_Legs.mp4
        │   └── ...
        ├── camera_1/
        │   ├── p0_cam1_seg000_Touch_Dashboard.mp4
        │   └── ...
        ├── camera_azure_color/
        │   ├── p0_camazure_color_seg000_Touch_Dashboard.mp4
        │   └── ...
        ├── camera_azure_depth/
        │   ├── p0_camazure_depth_seg000_Touch_Dashboard.mp4
        │   └── ...
        ├── camera_azure_ir/
        │   ├── p0_camazure_ir_seg000_Touch_Dashboard.mp4
        │   └── ...
        └── training_summary.csv
```

## 🔍 **Azure Kinect Processing**

The tool now processes Azure Kinect as **three separate cameras**:

- **`azure_color`**: Color video (1920x1080, 30 FPS)
- **`azure_depth`**: Depth video (512x512, 30 FPS) 
- **`azure_ir`**: Infrared video (512x512, 30 FPS)

All three use the same timestamp data from `webcam_azure_kinect.csv` for synchronization.

## 🚀 **Usage**

### Basic Usage

```bash
# Process videos for participant 0 with default settings
python post_processing.py 0

# Custom segment duration (20 seconds instead of 15)
python post_processing.py 0 --segment-duration 20

# Custom reading time cutoff (7 seconds instead of 5)
python post_processing.py 0 --reading-cutoff 7

# Minimum segment duration (5 seconds instead of 3)
python post_processing.py 0 --min-duration 5

# Statistics only (no video processing)
python post_processing.py 0 --stats-only
```

### Command Line Arguments

- `participant_id`: Participant ID (required)
- `--segment-duration`: Duration of each segment in seconds (default: 15)
- `--base-path`: Base path for dataset (default: "dataset")
- `--reading-cutoff`: Seconds to cut from start (reading time, default: 5)
- `--min-duration`: Minimum segment duration in seconds (default: 3)
- `--stats-only`: Only show statistics, don't process videos

## 📋 **Training Summary CSV**

The `training_summary.csv` contains:

| Column | Description |
|--------|-------------|
| `participant_id` | Participant identifier |
| `camera_id` | Camera identifier (e.g., 1, 2, azure_color, azure_depth, azure_ir) |
| `segment_id` | Unique segment identifier |
| `filename` | Generated filename for the video segment |
| `filepath` | Full path to the video segment |
| `start_time` | Segment start timestamp |
| `end_time` | Segment end timestamp |
| `gesture_name` | Name of the gesture |
| `gesture_index` | Gesture index from auto_labels |
| `gesture_time` | Exact gesture timestamp |
| `duration_seconds` | Duration of the segment |
| `training_duration` | Fixed training duration (15 seconds) |
| `reading_time_excluded` | Whether reading time was excluded |
| `training_ready` | Whether segment meets training criteria |

## 🎯 **Model Training Optimizations**

### **Video Quality Standards**
- **Frame Rate**: Standardized to 30 FPS for consistent training
- **Format**: MP4 with mp4v codec for compatibility
- **Resolution**: Preserves original resolution (typically 1920x1080)

### **Content Filtering**
- **Reading Time**: First 5 seconds excluded (participant reading instructions)
- **Segment Duration**: Fixed 15-second segments from gesture start
- **Minimum Duration**: 3-second minimum for meaningful training samples
- **Gesture Focus**: Segments start at gesture timestamp and extend 15 seconds forward

### **Naming Convention**
- **Format**: `p{participant_id}_cam{camera_id}_seg{index:03d}_{gesture_name}.mp4`
- **Examples**: 
  - `p0_cam0_seg000_Touch_Dashboard.mp4` (webcam)
  - `p0_camazure_color_seg000_Touch_Dashboard.mp4` (Azure Kinect color)
  - `p0_camazure_depth_seg000_Touch_Dashboard.mp4` (Azure Kinect depth)
  - `p0_camazure_ir_seg000_Touch_Dashboard.mp4` (Azure Kinect IR)

## 📊 **Example**

```python
from post_processing import PostProcessor

# Create processor for participant 0
processor = PostProcessor("0")

# Process videos with 15-second segments
processor.process_videos(segment_duration=15)

# Show training statistics
processor.get_processing_statistics()
```

## 🔧 **Configuration**

### **Training Parameters**
```python
# Default settings optimized for model training
reading_time_cutoff = 5      # Cut first 5 seconds (reading time)
min_segment_duration = 3     # Minimum 3 seconds for training
target_fps = 30             # Standardize to 30 FPS
segment_duration = 15       # Fixed 15-second segments
```

### **Custom Settings**
```bash
# For longer segments
python post_processing.py 0 --segment-duration 20

# For longer reading time
python post_processing.py 0 --reading-cutoff 7
```

## 📈 **Quality Assurance**

### **Automatic Filtering**
- ✅ Excludes segments during reading time
- ✅ Filters out segments shorter than minimum duration
- ✅ Merges overlapping segments intelligently
- ✅ Standardizes frame rate for consistent training

### **Training Readiness**
- ✅ All videos at consistent 30 FPS
- ✅ Proper naming convention for dataset organization
- ✅ Comprehensive summary CSV for training metadata
- ✅ Quality-filtered segments only

## 🚨 **Troubleshooting**

### **Common Issues**

1. **"No valid segments found for training"**
   - Check gesture timestamps overlap with video timestamps
   - Verify reading time cutoff isn't too aggressive
   - Ensure minimum duration isn't too high

2. **"No frames found in time window"**
   - Verify CSV timestamp format matches video timestamps
   - Check that gesture times fall within video duration

3. **"Video file not found"**
   - Ensure video files exist in expected locations
   - Check file naming convention (webcam_X.mp4)

### **Debug Mode**

Add debug output by modifying the script:

```python
# Add to PostProcessor.__init__()
self.debug = True

# Add debug prints in _extract_video_segment()
if self.debug:
    print(f"Processing {video_path}")
    print(f"Time window: {start_time} to {end_time}")
    print(f"Found {len(segment_frames)} frames")
    print(f"Reading time cutoff: {self.reading_time_cutoff}s")
```

## 📝 **Notes**

- **Reading Time**: First 5 seconds are automatically excluded as they contain instruction reading, not performance
- **Training Focus**: All segments are optimized for machine learning model training
- **Consistency**: All videos are standardized to 30 FPS for uniform training data
- **Quality**: Only segments meeting minimum duration requirements are included
- **Organization**: Clear naming convention for easy dataset management