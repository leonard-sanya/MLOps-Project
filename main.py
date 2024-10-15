from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from mysql.connector import connect
from passlib.context import CryptContext
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, INT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
import cv2
import face_recognition
import io
import numpy as np
import os
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8050"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = f"mysql+mysqldb://{db_username}:{db_password}@{db_host}:3306/{db_name}"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
Base.metadata.create_all(bind=engine)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    username = Column(String(10), unique=True, index=True)
    email = Column(String(35), unique=True, index=True)
    password = Column(String(100))
    face_encoding = Column(LargeBinary)
    is_admin = Column(INT)


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
    admin_name = "Administrator"
    admin_username = "admin"
    admin_password = "admin_password"

    db_user = db.query(User).filter(User.username == admin_username).first()
    if not db_user:
        hashed_password = hash_password(admin_password)
        admin_user = User(
            name=admin_name,
            username=admin_username,
            email="ignatiusboadi@gmail.com",
            password=hashed_password,
            is_admin=1
        )
        db.add(admin_user)
        db.commit()


create_admin_user()


@app.post("/enroll")
async def enroll(
        name: str = Form(...),
        username: str = Form(...),
        email: str = Form(...),
        password: str = Form(...),
        is_admin: int = Form(0),
        image: UploadFile = File()):

    if is_admin == 1:
        raise HTTPException(status_code=403, detail="Admin account cannot be enrolled.")

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

        user = User(
            name=name,
            username=username,
            email=email,
            password=hashed_password,
            is_admin=is_admin,
            face_encoding=face_encoding.tobytes()
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return {"message": "User enrolled successfully"}

    except HTTPException as e:
        raise e

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {type(e).__name__}")

    finally:
        db.close()


@app.delete("/unenroll")
async def unenroll_user(username: str = Form(...), token: str = Depends(oauth2_scheme)):
    db: Session = SessionLocal()
    try:
        current_user = decode_token(token)

        db_user = db.query(User).filter(User.username == current_user).first()
        if not db_user or db_user.is_admin != 1:
            raise HTTPException(status_code=403, detail="You are not authorized to perform this action. Only admins can delete users.")

        user_to_unenroll = db.query(User).filter(User.username == username).first()
        if not user_to_unenroll:
            raise HTTPException(status_code=404, detail="User not found.")

        db.delete(user_to_unenroll)
        db.commit()

        return {"message": f"{username} has been unenrolled successfully"}

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="An error occurred during unenrollment.")

    finally:
        db.close()


@app.put("/update")
async def update_user(
        name: str = Form(None),
        username: str = Form(None),
        email: str = Form(None),
        password: str = Form(None),
        token: str = Depends(oauth2_scheme)
):
    db: Session = SessionLocal()

    try:
        current_user = decode_token(token)
        db_user = db.query(User).filter(User.username == current_user).first()
        if not db_user or db_user.is_admin != 1:
            raise HTTPException(status_code=403,
                                detail="You are not authorized to perform this action. Only admins can perform updates.")

        user_to_update = db.query(User).filter(User.username == username).first()
        if not user_to_update:
            raise HTTPException(status_code=404, detail="Username not found")

        if name:
            user_to_update.name = name
        if username:
            user_to_update.username = username
        if email:
            user_to_update.email = email
        if password:
            user_to_update.password = hash_password(password)

        db.commit()
        return {"message": f"User {username} updated successfully"}

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {type(e).__name__}")

    finally:
        db.close()


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db: Session = SessionLocal()
    db_user = db.query(User).filter(User.username == form_data.username).first()
    name = db_user.name
    email = db_user.email
    if not db_user:
        # return {'message': 'User not enrolled.'}
        raise HTTPException(status_code=400, detail="User not enrolled")

    if not verify_password(form_data.password, db_user.password):
        # return {'message': 'Incorrect password.'}
        raise HTTPException(status_code=400, detail="Incorrect password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    yag = yagmail.SMTP('ammi.mlops.group1@gmail.com', 'pktwlpqogrkotiyg')
    message = f'''Dear {name.split()[0]},
                Kindly find below the bearer token for you to access.
                
                {access_token}

                Kindest regards,
                Group 1'''
    yag.send(email, f'Bearer token', message)

    return {"message": 'Token generated and sent to registered email.', "access_token": access_token,
            "token_type": "bearer"}


@app.post("/face_recognition")
async def face_recognition_endpoint(image: UploadFile = File()):
    db: Session = SessionLocal()

    try:
        image_bytes = await image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        if not face_locations:
            raise HTTPException(status_code=400, detail="No face detected in the image.")

        users = db.query(User).all()
        recognized_users = []

        for face_encoding in face_encodings:
            for user in users:
                if user.face_encoding is None:
                    continue
                stored_encoding = np.frombuffer(user.face_encoding, dtype=np.float64)
                matches = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.4)
                if matches[0]:
                    recognized_users.append(user.username)

        return {'message': 1} if len(recognized_users) else {'message': 0}

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {type(e).__name__}")

    finally:
        db.close()

