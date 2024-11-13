import os
import cv2
import json
import uuid
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# Define the model paths
growth_model_path = 'lettuce_growth_stage_detection(100).pt'
disease_model_path = 'lettuce_diease_detection(100).pt'

# Function to load the YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}.")
        return model
    except KeyError as e:
        print(f"KeyError: {e}. This might indicate an issue with the model file.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Load the models
growth_model = load_model(growth_model_path)
disease_model = load_model(disease_model_path)

@app.post("/detect-growth-stage")
async def detect_growth_stage(file: UploadFile = File(...)):
    return await detect_objects(file, growth_model)

@app.post("/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    return await detect_objects(file, disease_model)

async def detect_objects(file: UploadFile, model):
    # Read the uploaded image
    image = await file.read()
    image = Image.open(BytesIO(image))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Resize the image to 448x448
    image = cv2.resize(image, (448, 448))

    # Make predictions using the YOLO model
    results = model(image)
    print("Results: ", results, "/nType of results is ", type(results))

    # Assuming results[0] is your Results object
    result = results[0]

    # Create a dictionary with relevant attributes
    result_dict = {
        "boxes": result.boxes.data.numpy().tolist() if result.boxes.data is not None else None,
        "keypoints": result.keypoints.tolist() if result.keypoints is not None else None,
        "masks": result.masks.tolist() if result.masks is not None else None,
        "names": result.names,
        "orig_shape": result.orig_shape,
        "path": result.path,
        "speed": result.speed,
        "image_size": f"{image.shape[1]}x{image.shape[0]}",
        "detected_classes": [result.names[int(box[5])] for box in result.boxes.data.numpy()] if result.boxes.data is not None else []
    }

    # Convert the dictionary to a JSON string
    result_json = json.dumps(result_dict, indent=4)

    # Print the JSON string
    # print(result_json)

    # Draw annotations and save the image
    # save_path = draw_annotations(image, result)

    # Return the detection results as JSON along with the path to the saved image
    return JSONResponse(content={"results": result_dict, "saved_image_path": save_path}, media_type="application/json")

def draw_annotations(image, result):
    # Draw bounding boxes and labels on the image
    for box in result.boxes.data:
        x1, y1, x2, y2, conf, class_id = box.tolist()
        label = f"{result.names[int(class_id)]}: {conf:.2f}"
        # Draw the bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        # Put the label above the bounding box
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Create a directory to save the annotated images if it doesn't exist
    save_dir = 'annotated_images'
    os.makedirs(save_dir, exist_ok=True)

    # Generate a unique name for the annotated image
    unique_name = f"annotated_image_{uuid.uuid4()}.jpg"
    save_path = os.path.join(save_dir, unique_name)
    
    # Save the annotated image
    cv2.imwrite(save_path, image)

    return save_path