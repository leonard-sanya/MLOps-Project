import io

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Form
from mysql.connector import connect
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, INT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
import numpy as np
import os
import face_recognition
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
import yagmail

load_dotenv()
db_host = os.getenv('IP_ADDRESS')
db_name = os.getenv('DB_NAME')
db_username = os.getenv('DB_USERNAME')
db_password = os.getenv('DB_PASSWORD')
connection_name = os.getenv('CONNECTION_NAME')

conn = connect(
    host=db_host,
    user=db_username,
    password=db_password,
    database=db_name
)

cursor = conn.cursor()

app = FastAPI()

DATABASE_URL = f"mysql+mysqldb://{db_username}:{db_password}@{db_host}:3306/{db_name}"
print(DATABASE_URL)
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
    username = Column(String(10), unique=True, index=True)
    email = Column(String(35), unique=True, index=True)
    password = Column(String(100))
    face_encoding = Column(LargeBinary)
    is_admin = Column(INT)


Base.metadata.create_all(bind=engine)


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    face_encoding: bytes
    is_admin: int


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


def create_admin_user():
    db: Session = SessionLocal()
    admin_username = "admin"
    admin_password = "admin_password"

    db_user = db.query(User).filter(User.username == admin_username).first()
    if not db_user:
        hashed_password = hash_password(admin_password)
        admin_user = User(
            username=admin_username,
            email="admin@example.com",
            password=hashed_password,
            is_admin=1
        )
        db.add(admin_user)
        db.commit()
        print("Admin user created successfully.")


create_admin_user()


@app.post("/enroll")
async def enroll(
        username: str = Form(...),
        email: str = Form(...),
        password: str = Form(...),
        is_admin: int = Form(0),
        image: UploadFile = File()):
    if is_admin == 1:
        raise HTTPException(status_code=403, detail="Admin account cannot be enrolled.")
    print('requests received')
    db: Session = SessionLocal()

    try:
        db_user = db.query(User).filter(User.username == username).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Username already registered")

        image_data = await image.read()
        image_stream = io.BytesIO(image_data)
        pil_image = Image.open(image_stream)
        rgb_frame = np.array(pil_image)

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

    except Exception as e:
        return {'message': f'{type(e).__name__}'}


@app.delete("/unenroll")
async def unenroll_user(username: str=Form(...), token: str = Depends(oauth2_scheme)):
    db: Session = SessionLocal()
    current_user = decode_token(token)

    db_user = db.query(User).filter(User.username == current_user).first()
    if not db_user or db_user.is_admin != 1:
        raise HTTPException(status_code=403, detail="Not authorized to perform this action")

    user_to_unenroll = db.query(User).filter(User.username == username).first()
    if not user_to_unenroll:
        raise HTTPException(status_code=404, detail="User not found")

    user_to_unenroll.face_encoding = None
    db.commit()
    return {"message": f"{username} has been unenrolled successfully"}


@app.put("/user")
async def update_user(
        username: str = Form(...),
        email: str = Form(...),
        password: str = Form(None),
        token: str = Depends(oauth2_scheme)
):
    db: Session = SessionLocal()
    current_user = decode_token(token)

    db_user = db.query(User).filter(User.username == current_user).first()
    if not db_user or db_user.is_admin != 1:
        print('403')
        raise HTTPException(status_code=403, detail="Not authorized to perform this action")

    user_to_update = db.query(User).filter(User.username == username).first()
    if not user_to_update:
        print(404)
        raise HTTPException(status_code=404, detail="User not found")

    user_to_update.username = username
    user_to_update.email = email
    if password:
        user_to_update.password = hash_password(password)  # Hash new password

    db.commit()
    print('committed')
    return {
        "message": f"User {username} updated successfully",
        "user": {
            "username": user_to_update.username,
            "email": user_to_update.email,
            "is_admin": user_to_update.is_admin
        }
    }


#################################################################################
# USER LOG IN
#################################################################################

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db: Session = SessionLocal()
    db_user = db.query(User).filter(User.username == form_data.username).first()
    email = db_user.email
    print('token initialized')
    if not db_user:
        raise HTTPException(status_code=400, detail="User not enrolled")

    if not verify_password(form_data.password, db_user.password):
        raise HTTPException(status_code=400, detail="Incorrect password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    yag = yagmail.SMTP('ammi.mlops.group1@gmail.com', 'pktwlpqogrkotiyg')
    message = f'''Dear User,
                Kindly find below the bearer token for you to access
                {access_token}

                Kindest regards,
                Group 1'''
    yag.send(email, f'Bearer token', message)

    return {"access_token": access_token, "token_type": "bearer"}


#################################################################################
# USER FACE RECOGNITION
#################################################################################
@app.post("/face_recognition")
async def face_recognition_endpoint(image: UploadFile = File()):
    # Read the uploaded image file contents
    db: Session = SessionLocal()
    image_bytes = await image.read()

    # Convert the uploaded image bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Convert image to RGB format for face recognition
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    if not face_locations:
        return {"message": "No face detected in the image."}

    # Query the database to retrieve all users
    users = db.query(User).all()

    recognized_users = []

    for face_encoding in face_encodings:
        for user in users:
            if user.face_encoding is None:
                continue

            # Load the stored face encoding and compare with the current face encoding
            stored_encoding = np.frombuffer(user.face_encoding, dtype=np.float64)
            matches = face_recognition.compare_faces([stored_encoding], face_encoding)
            if matches[0]:
                recognized_users.append(user.username)

    # Use MTCNN to detect faces in the image
    # detector = MTCNN()
    # detections = detector.detect_faces(rgb_img)
    #
    # if not detections:
    #     raise HTTPException(status_code=400, detail="No faces detected in the image.")
    #
    # # Draw boxes and keypoints around the faces in the image
    # img_with_dets = rgb_img.copy()
    # min_conf = 0.9
    #
    # box_color = (255, 0, 0)
    # box_thickness = 3
    # dot_color = (0, 255, 0)
    # dot_size = 6
    #
    # for det in detections:
    #     if det['confidence'] >= min_conf:
    #         x, y, width, height = det['box']
    #         keypoints = det['keypoints']
    #
    #         cv2.rectangle(img_with_dets, (x, y), (x + width, y + height), box_color, box_thickness)
    #         cv2.circle(img_with_dets, keypoints['left_eye'], dot_size, dot_color, -1)
    #         cv2.circle(img_with_dets, keypoints['right_eye'], dot_size, dot_color, -1)
    #         cv2.circle(img_with_dets, keypoints['nose'], dot_size, dot_color, -1)
    #         cv2.circle(img_with_dets, keypoints['mouth_left'], dot_size, dot_color, -1)
    #         cv2.circle(img_with_dets, keypoints['mouth_right'], dot_size, dot_color, -1)
    #
    # # Encode the image with detections to JPEG format
    # _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_with_dets, cv2.COLOR_RGB2BGR))
    # img_bytes = BytesIO(img_encoded.tobytes())

    return {'recognized': 1} if len(recognized_users) else {'recognized': 0}
