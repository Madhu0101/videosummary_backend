import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import json
from datetime import datetime

class MongoDBInspector:
    def __init__(self, mongo_uri="mongodb://localhost:27017"):
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client["videosummary"]
        self.users_collection = self.db["users"]
        self.videos_collection = self.db["videos"]
        self.tokens_collection = self.db["tokens"]

    async def inspect_users(self):
        """Check all users in the database"""
        print("=== USERS COLLECTION ===")
        users = await self.users_collection.find({}).to_list(length=1000)
        print(f"Total users: {len(users)}")
        
        for user in users:
            print(f"\nUser ID: {user['_id']}")
            print(f"Username: {user.get('username', 'N/A')}")
            print(f"Email: {user.get('email', 'N/A')}")
            print(f"Created at: {user.get('created_at', 'N/A')}")
            print(f"Password hash exists: {'password_hash' in user}")
            if 'password_hash' in user:
                print(f"Password hash (first 20 chars): {user['password_hash'][:20]}...")

    async def inspect_tokens(self):
        """Check all tokens in the database"""
        print("\n=== TOKENS COLLECTION ===")
        tokens = await self.tokens_collection.find({}).to_list(length=1000)
        print(f"Total tokens: {len(tokens)}")
        
        for token in tokens:
            print(f"\nToken ID: {token.get('_id', 'N/A')}")
            print(f"User ID: {token.get('user_id', 'N/A')}")
            print(f"Token (first 20 chars): {token.get('token', 'N/A')[:20]}...")

    async def inspect_videos(self):
        """Check all videos in the database"""
        print("\n=== VIDEOS COLLECTION ===")
        videos = await self.videos_collection.find({}).to_list(length=1000)
        print(f"Total videos: {len(videos)}")
        
        for video in videos:
            print(f"\nVideo ID: {video['_id']}")
            print(f"User ID: {video.get('user_id', 'N/A')}")
            print(f"Filename: {video.get('original_filename', 'N/A')}")
            print(f"Status: {video.get('status', 'N/A')}")
            print(f"Summary Status: {video.get('summary_status', 'N/A')}")

    async def find_user_by_email(self, email):
        """Find a specific user by email"""
        print(f"\n=== SEARCHING FOR USER: {email} ===")
        user = await self.users_collection.find_one({"email": email})
        if user:
            print("User found!")
            print(f"User ID: {user['_id']}")
            print(f"Username: {user.get('username', 'N/A')}")
            print(f"Email: {user.get('email', 'N/A')}")
            print(f"Created at: {user.get('created_at', 'N/A')}")
            print(f"Password hash exists: {'password_hash' in user}")
            return user
        else:
            print("User not found!")
            return None

    async def clean_expired_tokens(self):
        """Clean up old tokens (optional maintenance)"""
        print("\n=== CLEANING EXPIRED TOKENS ===")
        result = await self.tokens_collection.delete_many({})
        print(f"Deleted {result.deleted_count} tokens")

    async def inspect_all(self):
        """Run all inspections"""
        await self.inspect_users()
        await self.inspect_tokens()
        await self.inspect_videos()

    async def close(self):
        """Close the database connection"""
        self.client.close()

# Usage example
async def main():
    inspector = MongoDBInspector()
    
    try:
        # Inspect all collections
        await inspector.inspect_all()
        
        # Search for a specific user (replace with actual email)
        # await inspector.find_user_by_email("user@example.com")
        
        # Clean expired tokens if needed
        # await inspector.clean_expired_tokens()
        
    finally:
        await inspector.close()

if __name__ == "__main__":
    asyncio.run(main())