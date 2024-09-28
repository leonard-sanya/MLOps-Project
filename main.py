from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status,Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from datetime import datetime, timedelta, timezone
from typing import Union
from jwt import PyJWTError
from passlib.context import CryptContext
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from fastapi.responses import FileResponse
from PIL import ImageFont, ImageDraw, Image
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

import dagshub
import jwt
import io
import mlflow
import os
import torch
import time
import timm
import torchvision.transforms as transforms

import pandas as pd


# JWT settings
SECRET_KEY = "fdb3e44ba75f4d770ee8de98e488bc3ebcf64dc3066c8140a1ae620c30964454"  # Replace with your own secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

os.environ['MLFLOW_TRACKING_USERNAME'] = 'ignatiusboadi'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '67ea7e8b48b9a51dd1748b8bb71906cc5806eb09'
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/ignatiusboadi/mlops-tasks.mlflow'

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Define the mapping of predicted class indices to names
class_mapping = {
    0: 'Angelina Jolie', 1: 'Brad Pitt', 2: 'Denzel Washington',
    3: 'Hugh Jackman', 4: 'Jennifer Lawrence', 5: 'Johnny Depp',
    6: 'Kate Winslet', 7: 'Leonardo DiCaprio', 8: 'Megan Fox',
    9: 'Natalie Portman', 10: 'Nicole Kidman', 11: 'Robert Downey Jr',
    12: 'Sandra Bullock', 13: 'Scarlett Johansson', 14: 'Tom Cruise',
    15: 'Tom Hanks', 16: 'Will Smith'
}

# User database (mock)
users_db = {
    "admin": {"username": "admin", "password": pwd_context.hash("adminpass"), "role": "admin"},
    "user": {"username": "user", "password": pwd_context.hash("userpass"), "role": "user"},
}

# Load the face detection model (YOLOv8)
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
face_model = YOLO(model_path)

# Load the face classification model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("rexnet_150", pretrained=False, num_classes=len(class_mapping))
model.load_state_dict(torch.load('./models/faces_best_model.pth', map_location=device))
model.eval()
model.to(device)

# Define image transformation for classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
])

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Welcome to the Face Recognition API"}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


dagshub.init(repo_owner='ignatiusboadi', repo_name='mlops-tasks', mlflow=True)


# Helper functions
def start_or_get_run():
    if mlflow.active_run() is None:
        mlflow.start_run()
    else:
        print(f"Active run with UUID {mlflow.active_run().info.run_id} already exists")


def end_active_run():
    if mlflow.active_run() is not None:
        mlflow.end_run()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user or not verify_password(password, user["password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return username
    except PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Celebrity-face-recognition")


# Token Generation Endpoint
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/predict/{door_number}")
async def face_recognition(door_number, file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    end_active_run()
    start_or_get_run()
    # Load the image
    image = Image.open(file.file).convert("RGB")
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)
    mlflow.log_param("image_size", image.size)

    start_time = time.time()

    results = face_model(image)

    if not results or len(results[0].boxes) == 0:
        raise HTTPException(status_code=400, detail="No faces detected in the image.")

    try:
        font_path = "/Library/Fonts/Arial.ttf"  # Update this path according to your system
        font = ImageFont.truetype(font_path, size=24)
    except IOError:
        font = ImageFont.load_default()  

    # Process only if faces are detected
    draw = ImageDraw.Draw(image)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        
        # Crop the detected face from the image
        face_image = image.crop((x1, y1, x2, y2))
        face_tensor = transform(face_image).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get the raw model output (logits)
            output = model(face_tensor)
            
            output = torch.exp(output - torch.max(output))  
            output = output / output.sum(dim=1, keepdim=True) 
            predicted_class = torch.argmax(output, dim=1).item()
            mlflow.log_param('model_output', class_mapping.get(predicted_class))
            confidence, _ = torch.max(output, dim=1)
            confidence = confidence.item()
            mlflow.log_metric("confidence", confidence)
            mlflow.log_param('predicted_class', predicted_class)
            # mlflow.log_param("predicted_class", class_mapping.get(predicted_class, "Unknown"))

        if confidence < 0.95:
            predicted_name = "Unknown Face"
        else:
            predicted_name = class_mapping.get(predicted_class)
        mlflow.log_param('door number', door_number)
        mlflow.log_param('predicted_name', predicted_name)
        # Draw bounding box and label on the original image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), predicted_name, fill="white", font=font)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    execution_time = time.time() - start_time
    mlflow.log_metric("execution_time", execution_time)

    if int(door_number) == predicted_class:
        action = 'Flow to open door initialized'
    else:
        action = 'Access denied. Please try again or contact security.'
    mlflow.log_param('Action', action)
    mlflow.end_run()
    return action

#     return StreamingResponse(
#     img_byte_arr,
#     media_type="image/jpeg",
#     headers={
#         "Predicted-Door-Num": str(predicted_class),
#         "X-Confidence": str(confidence)
#     }
# )


