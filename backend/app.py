from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional
import io
from ultralytics import YOLO
from PIL import Image
import uvicorn

app = FastAPI(title="YOLO Fruit Detection API")

# Load the YOLO model (make sure the path is correct)
model = YOLO("backend/yolo11n.pt")

# Custom class names with translations
custom_class_names = {
    46: "Banana (কলা)"
}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.3

# Global variable to store latest detected fruits
latest_detected_fruits = []

@app.post("/detect")
async def detect_fruit(image_file: Optional[UploadFile] = File(None)):
    global latest_detected_fruits

    # Handle missing file
    if not image_file:
        return JSONResponse(content={
            "status": False,
            "message": "Upload image file",
            "data": None
        }, status_code=400)

    try:
        # Read and validate image
        contents = await image_file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception:
            return JSONResponse(content={
                "status": False,
                "message": "Upload image file",
                "data": None
            }, status_code=400)

        # Run YOLO inference
        try:
            results = model(image)
        except Exception:
            return JSONResponse(content={
                "status": False,
                "message": "Model inference failed",
                "data": None
            }, status_code=500)

        # Parse detected fruits
        fruit_names = set()
        for result in results:
            for box in result.boxes:
                if float(box.conf) >= CONFIDENCE_THRESHOLD:
                    class_id = int(box.cls)
                    if class_id in custom_class_names:
                        fruit_names.add(custom_class_names[class_id])

        # Sort and store detected fruit names
        detected_fruits = sorted(fruit_names, key=lambda x: x[0])
        latest_detected_fruits = detected_fruits
        fruits_string = ", ".join(detected_fruits) if detected_fruits else None

        # Return response
        return JSONResponse(content={
            "status": True,
            "message": "Successful" if fruits_string else "Not found",
            "data": fruits_string
        })

    except Exception as e:
        # Catch-all for unexpected server errors
        latest_detected_fruits = []
        return JSONResponse(content={
            "status": False,
            "message": str(e),
            "data": None
        }, status_code=500)

@app.get("/fruit-names")
async def get_detected_fruit_names():
    fruits_string = ", ".join(latest_detected_fruits) if latest_detected_fruits else None
    return JSONResponse(content={
        "status": True,
        "message": "Successful" if fruits_string else "Not found",
        "data": fruits_string
    })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
