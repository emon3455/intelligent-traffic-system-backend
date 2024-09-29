import cv2
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

model = YOLO('yolov8s.pt')

class_names = model.names

target_classes = ["person", "bicycle", "car", "motorbike", "bus", "truck"]

@app.get("/")
def home():
    return JSONResponse("Object Detection Server Is Running")


@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):

    temp_video_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(temp_video_path, "wb") as f:
        f.write(await file.read())

    video_capture = cv2.VideoCapture(temp_video_path)

    fps = video_capture.get(cv2.CAP_PROP_FPS)

    object_counts = {}

    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % int(fps) == 0:
            results = model(frame)

            for detection in results[0].boxes:
                class_id = int(detection.cls)
                class_name = class_names[class_id]

                if class_name in target_classes:
                    if class_name in object_counts:
                        object_counts[class_name] += 1
                    else:
                        object_counts[class_name] = 1

        frame_count += 1

    video_capture.release()
    os.remove(temp_video_path)

    return JSONResponse(content={"object_counts": object_counts})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
