from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Form
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
import cv2
import numpy as np
import os
import cv2
# import face_recognition
import time
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
from mtcnn import MTCNN
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
import cv2

load_dotenv()
ip_address = os.getenv('IP_ADDRESS')
db_name = os.getenv('DB_NAME')
db_username = os.getenv('DB_USERNAME')
db_password = os.getenv('DB_PASSWORD')
connection_name = os.getenv('CONNECTION_NAME')


app = FastAPI()

DATABASE_URL = "sqlite:///./test.db"

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    face_encoding = Column(LargeBinary)


Base.metadata.create_all(bind=engine)


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    face_encoding: bytes


class UserLogin(BaseModel):
    username: str
    password: str


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


#################################################################################
# USER ENROLLMENT
#################################################################################

@app.post("/enroll")
async def enroll(
        username: str = Form(...),
        email: str = Form(...),
        password: str = Form(...)
):
    db: Session = SessionLocal()
    video_capture = None

    try:
        db_user = db.query(User).filter(User.username == username).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Username already registered")

        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            raise HTTPException(status_code=500, detail="Could not access the camera.")

        print("Please focus on the camera. The image will be captured in 5 seconds...")
        time.sleep(5)

        ret, frame = video_capture.read()
        if not ret:
            raise HTTPException(status_code=500, detail="Could not capture an image.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face detected in the image.")

        face_encoding = face_encodings[0]

        hashed_password = hash_password(password)

        user = User(username=username, email=email, password=hashed_password, face_encoding=face_encoding.tobytes())
        db.add(user)
        db.commit()
        db.refresh(user)

        return {"message": "User enrolled successfully"}

    finally:
        if video_capture is not None and video_capture.isOpened():
            video_capture.release()
        cv2.destroyAllWindows()


#################################################################################
# USER LOG IN
#################################################################################

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db: Session = SessionLocal()
    db_user = db.query(User).filter(User.username == form_data.username).first()

    if not db_user:
        raise HTTPException(status_code=400, detail="User not enrolled")

    if not verify_password(form_data.password, db_user.password):
        raise HTTPException(status_code=400, detail="Incorrect password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


#################################################################################
# USER FACE RECOGNITION
#################################################################################
@app.post("/face_recognition")
async def face_recognition_endpoint(token: str = Depends(oauth2_scheme)):
    video_capture = cv2.VideoCapture(0)

    print("Please focus on the camera. The image will be captured in 5 seconds...")
    time.sleep(5)

    ret, frame = video_capture.read()

    if not ret:
        raise HTTPException(status_code=500, detail="Could not access the camera.")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        raise HTTPException(status_code=400, detail="No faces detected in the image.")

    db: Session = SessionLocal()
    users = db.query(User).all()

    recognized_users = []

    for face_encoding in face_encodings:
        for user in users:
            stored_encoding = np.frombuffer(user.face_encoding, dtype=np.float64)
            matches = face_recognition.compare_faces([stored_encoding], face_encoding)

            if matches[0]:
                recognized_users.append(user.username)

    detector = MTCNN()
    detections = detector.detect_faces(frame)

    if not detections:
        raise HTTPException(status_code=400, detail="No faces detected in the image.")

    img_with_dets = frame.copy()
    min_conf = 0.9

    box_color = (255, 0, 0)
    box_thickness = 3
    dot_color = (0, 255, 0)
    dot_size = 6

    for det in detections:
        if det['confidence'] >= min_conf:
            x, y, width, height = det['box']
            keypoints = det['keypoints']

            cv2.rectangle(img_with_dets, (x, y), (x + width, y + height), box_color, box_thickness)

            cv2.circle(img_with_dets, keypoints['left_eye'], dot_size, dot_color, -1)
            cv2.circle(img_with_dets, keypoints['right_eye'], dot_size, dot_color, -1)
            cv2.circle(img_with_dets, keypoints['nose'], dot_size, dot_color, -1)
            cv2.circle(img_with_dets, keypoints['mouth_left'], dot_size, dot_color, -1)
            cv2.circle(img_with_dets, keypoints['mouth_right'], dot_size, dot_color, -1)

    _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_with_dets, cv2.COLOR_RGB2BGR))
    img_bytes = BytesIO(img_encoded.tobytes())

    video_capture.release()
    cv2.destroyAllWindows()

    return StreamingResponse(img_bytes, media_type="image/jpeg",
                             headers={"X-Recognized-Users": ', '.join(recognized_users)})

#################################################################################
# PREVIOUS MAIN.PY FILE
#################################################################################

# from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Header
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from fastapi.responses import StreamingResponse
# from datetime import datetime, timedelta, timezone
# from typing import Union
# from jwt import PyJWTError
# from passlib.context import CryptContext
# from huggingface_hub import hf_hub_download
# from ultralytics import YOLO
# from fastapi.responses import FileResponse
# from PIL import ImageFont, ImageDraw, Image
# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset

# import dagshub
# import jwt
# import io
# import mlflow
# import os
# import torch
# import time
# import timm
# import torchvision.transforms as transforms

# import pandas as pd

# SECRET_KEY = "fdb3e44ba75f4d770ee8de98e488bc3ebcf64dc3066c8140a1ae620c30964454"  # Replace with your own secret key
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 30

# os.environ['MLFLOW_TRACKING_USERNAME'] = 'ignatiusboadi'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = '67ea7e8b48b9a51dd1748b8bb71906cc5806eb09'
# os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/ignatiusboadi/mlops-tasks.mlflow'

# # Password hashing context
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # Define the mapping of predicted class indices to names
# class_mapping = {
#     0: 'Angelina Jolie', 1: 'Brad Pitt', 2: 'Denzel Washington',
#     3: 'Hugh Jackman', 4: 'Jennifer Lawrence', 5: 'Johnny Depp',
#     6: 'Kate Winslet', 7: 'Leonardo DiCaprio', 8: 'Megan Fox',
#     9: 'Natalie Portman', 10: 'Nicole Kidman', 11: 'Robert Downey Jr',
#     12: 'Sandra Bullock', 13: 'Scarlett Johansson', 14: 'Tom Cruise',
#     15: 'Tom Hanks', 16: 'Will Smith'
# }

# # User database (mock)
# users_db = {
#     "admin": {"username": "admin", "password": pwd_context.hash("adminpass"), "role": "admin"},
#     "user": {"username": "user", "password": pwd_context.hash("userpass"), "role": "user"},
# }

# # Load the face detection model (YOLOv8)
# model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
# face_model = YOLO(model_path)

# # Load the face classification model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = timm.create_model("rexnet_150", pretrained=False, num_classes=len(class_mapping))
# model.load_state_dict(torch.load('./models/faces_best_model.pth', map_location=device))
# model.eval()
# model.to(device)

# # Define image transformation for classification
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Welcome to the Face Recognition API"}


# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# dagshub.init(repo_owner='ignatiusboadi', repo_name='mlops-tasks', mlflow=True)


# # Helper functions
# def start_or_get_run():
#     if mlflow.active_run() is None:
#         mlflow.start_run()
#     else:
#         print(f"Active run with UUID {mlflow.active_run().info.run_id} already exists")


# def end_active_run():
#     if mlflow.active_run() is not None:
#         mlflow.end_run()


# def verify_password(plain_password, hashed_password):
#     return pwd_context.verify(plain_password, hashed_password)


# def authenticate_user(username: str, password: str):
#     user = users_db.get(username)
#     if not user or not verify_password(password, user["password"]):
#         return False
#     return user


# def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.now(timezone.utc) + expires_delta
#     else:
#         expire = datetime.now(timezone.utc) + timedelta(minutes=15)
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt


# def decode_token(token: str):
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
#         return username
#     except PyJWTError:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# mlflow.set_experiment("Celebrity-face-recognition")


# # Token Generation Endpoint
# @app.post("/token")
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
#     user = authenticate_user(form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": user["username"]}, expires_delta=access_token_expires
#     )
#     return {"access_token": access_token, "token_type": "bearer"}


# @app.post("/predict/{door_number}")
# async def face_recognition(door_number, file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
#     end_active_run()
#     start_or_get_run()
#     # Load the image
#     image = Image.open(file.file).convert("RGB")
#     to_tensor = transforms.ToTensor()
#     image_tensor = to_tensor(image)
#     mlflow.log_param("image_size", image.size)

#     start_time = time.time()

#     results = face_model(image)

#     if not results or len(results[0].boxes) == 0:
#         raise HTTPException(status_code=400, detail="No faces detected in the image.")

#     try:
#         font_path = "/Library/Fonts/Arial.ttf"  # Update this path according to your system
#         font = ImageFont.truetype(font_path, size=24)
#     except IOError:
#         font = ImageFont.load_default()

#         # Process only if faces are detected
#     draw = ImageDraw.Draw(image)
#     for box in results[0].boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])

#         # Crop the detected face from the image
#         face_image = image.crop((x1, y1, x2, y2))
#         face_tensor = transform(face_image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             # Get the raw model output (logits)
#             output = model(face_tensor)

#             output = torch.exp(output - torch.max(output))
#             output = output / output.sum(dim=1, keepdim=True)
#             predicted_class = torch.argmax(output, dim=1).item()
#             mlflow.log_param('model_output', class_mapping.get(predicted_class))
#             confidence, _ = torch.max(output, dim=1)
#             confidence = confidence.item()
#             mlflow.log_metric("confidence", confidence)
#             mlflow.log_param('predicted_class', predicted_class)
#             # mlflow.log_param("predicted_class", class_mapping.get(predicted_class, "Unknown"))

#         if confidence < 0.95:
#             predicted_name = "Unknown Face"
#         else:
#             predicted_name = class_mapping.get(predicted_class)
#         mlflow.log_param('door number', door_number)
#         mlflow.log_param('predicted_name', predicted_name)
#         # Draw bounding box and label on the original image
#         draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
#         draw.text((x1, y1), predicted_name, fill="white", font=font)

#     img_byte_arr = io.BytesIO()
#     image.save(img_byte_arr, format='JPEG')
#     img_byte_arr.seek(0)
#     execution_time = time.time() - start_time
#     mlflow.log_metric("execution_time", execution_time)

#     if int(door_number) == predicted_class:
#         action = 'Flow to open door initialized'
#     else:
#         action = 'Access denied. Please try again or contact security.'
#     mlflow.log_param('Action', action)
#     mlflow.end_run()
#     return action
