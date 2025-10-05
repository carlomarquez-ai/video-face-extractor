# Video Face Extractor

A Python tool for extracting faces from video files using deep learning models.

## Features
- Extracts faces from video frames
- Supports batch processing of videos
- Configurable face detection models
- Helper utilities for video and image processing

## Project Structure
- `main.py`: Entry point for running the application
- `video_processing.py`: Video frame extraction and processing logic
- `face_model.py`: Face detection model loading and inference
- `helpers.py`: Utility functions for file and image operations
- `config.py`: Configuration settings

## Requirements
- Python 3.7+
- OpenCV
- NumPy
- (Add any other dependencies used in your code)

## Installation
1. Clone this repository:
   ```sh
   git clone <repo-url>
   cd video-face-extractor
   ```
2. (Optional) Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the main script with your video file as input:
```sh
python main.py --input path/to/video.mp4 --output path/to/output_dir
```

Refer to the script's help for more options:
```sh
python main.py --help
```

## Configuration
Modify `config.py` to adjust detection thresholds, model paths, and other settings.

## License
MIT License
