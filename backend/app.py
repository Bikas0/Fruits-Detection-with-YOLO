from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from ultralytics import YOLO
from PIL import Image
import uvicorn

app = FastAPI(title="YOLO Fruit Detection API")

# Load the YOLO model
model = YOLO("backend/yolo11n.pt")

# Custom class names with translations
custom_class_names = {
    # Uncomment or add more if you want
    # 0: "Apple (আপেল)",
    46: "Banana (কলা)"
    # 2: "Mango (আম)",
    # 3: "Orange (কমলা)",
    # 4: "Pomegranate (আনারস)",
    # 5: "Watermelon (তরমুজ)"
}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.3

# Global variable to store latest detected fruits
latest_detected_fruits = []

@app.post("/detect")
async def detect_fruit(image_file: UploadFile = File(...)):
    global latest_detected_fruits

    contents = await image_file.read()
    
    try:
        image = Image.open(io.BytesIO(contents))
        results = model(image)
        
        fruit_names = set()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf)
                if confidence >= CONFIDENCE_THRESHOLD:
                    class_id = int(box.cls)
                    # Only add if class_id exists in custom_class_names
                    if class_id in custom_class_names:
                        fruit_names.add(custom_class_names[class_id])

        detected_fruits = sorted(fruit_names, key=lambda x: x[0])
        latest_detected_fruits = detected_fruits

        if not detected_fruits:
            return JSONResponse(
                content={"message": "Fruits not found."},
                status_code=200
            )

        # Convert list to comma-separated string
        fruits_string = ", ".join(detected_fruits)

        return JSONResponse(
            content={"fruits_name": fruits_string},
            status_code=200
        )

    except Exception as e:
        latest_detected_fruits = []
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/fruit-names")
async def get_detected_fruit_names():
    if not latest_detected_fruits:
        return JSONResponse(content={"message": "No fruits detected yet."}, status_code=200)

    fruits_string = ", ".join(latest_detected_fruits)
    return JSONResponse(content={"fruits_name": fruits_string}, status_code=200)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
