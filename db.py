import os
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client["videosummary"]
users_collection = db["users"]
videos_collection = db["videos"]
tokens_collection = db["tokens"]