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
async def upload_frame(file: UploadFile = File(...)):
    # Save the uploaded image frame to a temporary file
    temp_image_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(temp_image_path, "wb") as f:
        f.write(await file.read())

    # Load the image frame using OpenCV
    frame = cv2.imread(temp_image_path)

    # Run YOLO object detection on the current frame
    results = model(frame)

    # Count the number of detected objects in the frame
    object_count_in_frame = len(results[0].boxes)  # YOLOv8 structure for detections

    # Remove the temporary image file
    os.remove(temp_image_path)

    return JSONResponse(content={"total_object_count": object_count_in_frame})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
