import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Path to store the current signal state
state_file_path = 'signal_state.txt'


def read_signal_state():
    if os.path.exists(state_file_path):
        with open(state_file_path, 'r') as file:
            return file.read().strip()
    return "left"  # Default to left if file doesn't exist


def write_signal_state(state):
    with open(state_file_path, 'w') as file:
        file.write(state)


def detect_objects(frame_path):
    # Load the image frame using OpenCV
    frame = cv2.imread(frame_path)
    # Run YOLO object detection on the current frame
    results = model(frame)
    # Count the number of detected objects
    return len(results[0].boxes)  # YOLOv8 structure for detections


@app.post("/upload-video/")
async def upload_video(left_frame: UploadFile = File(...), right_frame: UploadFile = File(...)):
    # Save the uploaded files
    left_frame_path = os.path.join(tempfile.gettempdir(), "left_frame.jpg")
    right_frame_path = os.path.join(tempfile.gettempdir(), "right_frame.jpg")

    with open(left_frame_path, "wb") as buffer:
        shutil.copyfileobj(left_frame.file, buffer)

    with open(right_frame_path, "wb") as buffer:
        shutil.copyfileobj(right_frame.file, buffer)

    # Read the current signal state
    current_state = read_signal_state()

    # Detect objects in both frames
    left_has_objects = detect_objects(left_frame_path) > 0
    right_has_objects = detect_objects(right_frame_path) > 0

    # Determine new state based on logic
    if current_state == "left" and not left_has_objects:
        new_state = "right"  # Switch to right if left has no objects
    elif current_state == "right" and not right_has_objects:
        new_state = "left"  # Switch to left if right has no objects
    else:
        new_state = current_state  # Stay in the same state

    # Update the state file
    write_signal_state(new_state)

    # Prepare response
    left_signal = "play" if new_state == "left" else "stop"
    right_signal = "play" if new_state == "right" else "stop"

    return {"left_signal": left_signal, "right_signal": right_signal}


@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <h1>Traffic Signal Control API</h1>
    <p>Use the POST /upload-video/ endpoint to upload frames.</p>
    """
    return content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
