import os
from motor.motor_asyncio import AsyncIOMotorClient

# Connection URI
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://madhumithavenkatachalam:Imi5U8cBclbBfbza@cluster0.oyrzuku.mongodb.net/videosummary?retryWrites=true&w=majority"
)

# Create MongoDB client
mongo_client = AsyncIOMotorClient(MONGO_URI)

# Select database and collections
db = mongo_client["videosummary"]
users_collection = db["users"]
videos_collection = db["videos"]
tokens_collection = db["tokens"]
