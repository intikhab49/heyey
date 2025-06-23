"""
User model for authentication
"""
from tortoise.models import Model
from tortoise import fields
from passlib.context import CryptContext
from passlib.hash import bcrypt
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Model):
    """User model for authentication and authorization"""
    
    id = fields.IntField(pk=True)
    username = fields.CharField(max_length=50, unique=True)
    email = fields.CharField(max_length=100, unique=True)
    hashed_password = fields.CharField(max_length=128)
    is_active = fields.BooleanField(default=True)
    is_admin = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    last_login = fields.DatetimeField(null=True)
    
    class Meta:
        table = "users"
        
    def __str__(self):
        return self.username
    
    @classmethod
    async def create_user(cls, username: str, email: str, password: str, is_admin: bool = False):
        """Create a new user with hashed password"""
        try:
            hashed_password = pwd_context.hash(password)
            user = await cls.create(
                username=username,
                email=email,
                hashed_password=hashed_password,
                is_admin=is_admin
            )
            logger.info(f"Created user: {username}")
            return user
        except Exception as e:
            logger.error(f"Error creating user {username}: {str(e)}")
            raise
    
    @classmethod
    async def authenticate(cls, username: str, password: str):
        """Authenticate user with username and password"""
        try:
            user = await cls.get_or_none(username=username, is_active=True)
            if user and pwd_context.verify(password, user.hashed_password):
                # Update last login
                user.last_login = datetime.utcnow()
                await user.save()
                logger.info(f"User authenticated: {username}")
                return user
            return None
        except Exception as e:
            logger.error(f"Error authenticating user {username}: {str(e)}")
            return None
    
    @classmethod
    async def get_by_username(cls, username: str):
        """Get user by username"""
        try:
            return await cls.get_or_none(username=username)
        except Exception as e:
            logger.error(f"Error getting user {username}: {str(e)}")
            return None
    
    @classmethod
    async def get_by_email(cls, email: str):
        """Get user by email"""
        try:
            return await cls.get_or_none(email=email)
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {str(e)}")
            return None
    
    async def change_password(self, new_password: str):
        """Change user password"""
        try:
            self.hashed_password = pwd_context.hash(new_password)
            await self.save()
            logger.info(f"Password changed for user: {self.username}")
            return True
        except Exception as e:
            logger.error(f"Error changing password for user {self.username}: {str(e)}")
            return False
    
    async def deactivate(self):
        """Deactivate user account"""
        try:
            self.is_active = False
            await self.save()
            logger.info(f"User deactivated: {self.username}")
            return True
        except Exception as e:
            logger.error(f"Error deactivating user {self.username}: {str(e)}")
            return False
    
    def to_dict(self):
        """Convert user to dictionary (without password)"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }

# Utility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Get password hash"""
    return pwd_context.hash(password)
