import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import os

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

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    # Save the uploaded video to a temporary file
    temp_video_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(temp_video_path, "wb") as f:
        f.write(await file.read())

    # Open the video with OpenCV
    video_capture = cv2.VideoCapture(temp_video_path)

    total_object_count = 0

    # Process each frame of the video
    frameCount = 0
    while frameCount < 5:
        ret, frame = video_capture.read()
        if not ret:
            break  # Stop when there are no more frames

        # Run YOLO object detection on the current frame
        results = model(frame)

        # Count the number of detected objects in the current frame
        object_count_in_frame = len(results[0].boxes)  # YOLOv8 structure for detections
        total_object_count += object_count_in_frame
        frameCount = frameCount + 1

    video_capture.release()  # Release the video capture

    # Remove the temporary file
    os.remove(temp_video_path)

    return JSONResponse(content={"total_object_count": total_object_count})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
