from fastapi import FastAPI, Response, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi_jwt import JwtAuthorizationCredentials, JwtAccessBearer
from passlib.context import CryptContext
from datetime import timedelta, time
import datetime
import os
import dotenv
from models import Vebinar, User, Feed
from database import DataBase

dotenv.load_dotenv()

app = FastAPI()

database = DataBase()
pwd_context = CryptContext(schemes="bcrypt", deprecated="auto")
access_security = JwtAccessBearer(
    secret_key=os.getenv("JWT_SECRET"), auto_error=True)


def verify_password(plain_password: str, hash_password: str):
    return pwd_context.verify(plain_password, hash_password)


def get_hash_password(plain_password: str):
    return pwd_context.hash(plain_password)


app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],)


@app.post('/login')
async def login(user_auth: User, response: Response):
    try:
        subject = {"username": user_auth.username}
        access_token = access_security.create_access_token(
            subject=subject, expires_delta=timedelta(minutes=float(os.getenv("TOKEN_EXPIRES"))))
        access_security.set_access_cookie(response, access_token)
        return {"access_token": access_token}
    except Exception as e:
        print(e)
        raise HTTPException(403, 'сьебался отсюда быстро кабанчиком')


@app.get("/user/me")
async def get_me(credentials: JwtAuthorizationCredentials = Security(access_security)):
    return credentials.subject


@app.post("/user/add")
async def add_user(user: User, vebinar_id: int,  credentials: JwtAuthorizationCredentials = Security(access_security)):
    return database.add_user(user.username, user.hashed_pass, vebinar_id)


@app.get("/user/get_user/{id}")
async def get_user_by_id(id: int):
    result = database.get_user_by_id(id)
    return User(username=result[1], hashed_pass=result[2])


@app.post("/vebinar/add")
async def add_vebinar(vebinar: Vebinar, user_id: int, credentials: JwtAuthorizationCredentials = Security(access_security)):
    return database.add_vebinar(vebinar.name, user_id)


@app.get("/vebinar/get_vebinar/{id}")
async def get_vebinar_by_id(id: int):
    result = database.get_vebinar_by_id(id)
    return Vebinar(name=result[0])


@app.post("/feed/add_feed")
async def add_feed(feed: Feed, credentials: JwtAuthorizationCredentials = Security(access_security)):
    return database.add_feed(feed.feedback, feed.positive, feed.object, feed.is_relevant, feed.keypoint, feed.vebinar_id, feed.user_id)


@app.get("/feed/get_feed/{id}")
async def get_feed_by_id(id: int):
    result = database.get_feed_by_id(id)
    return Feed(feedback=result[1], positive=result[2], object=result[3], is_relevant=result[4], keypoint=result[5], vebinar_id=result[6], user_id=result[7])


@app.get("/vebinar/get_all")
async def get_all_feeds():
    result = database.get_vebinar_all()
    return [Vebinar(name=a[1]) for a in result]


@app.get("/feed/from_user/{user_id}")
async def get_feeds_from_user(user_id: int):

    result = database.get_feeds_from_one_user(user_id)
    print(result)
    return [Feed(feedback=a[1], positive=a[2], object=a[3], is_relevant=a[4], keypoint=a[5], vebinar_id=a[6], user_id=a[7]) for a in result]


@app.get("/feed/for_vebinar/{vebinar_id}")
async def get_feeds_for_vebinar(vebinar_id: int):
    result = database.get_feeds_for_one_vebinar(vebinar_id)
    return [Feed(feedback=a[1], positive=a[2], object=a[3], is_relevant=a[4], keypoint=a[5], vebinar_id=a[6], user_id=a[7]) for a in result]
