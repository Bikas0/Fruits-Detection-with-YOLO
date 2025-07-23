from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from ultralytics import YOLO
from PIL import Image
import uvicorn

app = FastAPI(title="YOLO Fruit Detection API")

# Load the YOLO model
model = YOLO("best.pt")

# Custom class names with translations
custom_class_names = {
    0: "Apple (আপেল)",
    1: "Banana (কলা)",
    2: "Mango (আম)",
    3: "Orange (কমলা)",
    4: "Pomegranate (আনারস)",
    5: "Watermelon (তরমুজ)"
}

# Confidence threshold (65%)
CONFIDENCE_THRESHOLD = 0.3

@app.post("/detect")
async def detect_fruit(image_file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await image_file.read()
    
    try:
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Perform detection
        results = model(image)
        
        # Process results with confidence threshold
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf)
                print(confidence)
                if confidence >= CONFIDENCE_THRESHOLD:  # Only consider detections above threshold
                    class_id = int(box.cls)
                    class_name = custom_class_names.get(class_id, "Unknown")
                    
                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": f"{confidence:.2f}%"
                    })
        
        if not detections:
            return JSONResponse(
                content={"message": "Fruits not found."},
                status_code=200
            )
        
        # Get all detected fruits (no longer just the highest confidence one)
        detected_fruits = list(set([d["class_name"] for d in detections]))
        detected_fruits = sorted(detected_fruits, key=lambda x: x[0])
        
        return JSONResponse(
            content={
                "fruits_name": detected_fruits,  
                "all_fruits": detections,       
                "Confidence_Used": CONFIDENCE_THRESHOLD
            },
            status_code=200
        )
    
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)