from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Body, Header
from pydantic import BaseModel, EmailStr
from models.user import User, pwd_context
import jwt
from simple_config import settings
from typing import Optional, List

router = APIRouter()

class UserCreate(BaseModel):
    username: str
    password: str
    email: EmailStr
    
    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    username: str
    password: str
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    
    class Config:
        from_attributes = True

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_admin: bool
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    trades: int
    balance: str
    
    class Config:
        from_attributes = True

# Dependency to get current user from JWT token
async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )
    
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user = await User.filter(username=username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user

# Dependency to get current admin user
async def get_current_admin_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

@router.post('/register', status_code=status.HTTP_201_CREATED, response_model=Token)
async def register(user: UserCreate):
    """Register a new user"""
    try:
        # Check if username already exists
        existing_user = await User.filter(username=user.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists
        existing_email = await User.filter(email=user.email).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = pwd_context.hash(user.password)
        new_user = await User.create(
            username=user.username,
            email=user.email,
            password_hash=hashed_password,
            last_login=datetime.now(),
            trades=0,
            balance="0 BTC"
        )
        
        # Generate token for immediate login
        token = jwt.encode({"sub": new_user.username}, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "username": new_user.username
        }
    except Exception as e:
        print(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post('/login', response_model=Token)
async def login(user: UserLogin):
    """Login user and return JWT token"""
    try:
        # Find user
        db_user = await User.filter(username=user.username).first()
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Verify password
        if not db_user.verify_password(user.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Update last login time
        db_user.last_login = datetime.now()
        await db_user.save()
        
        # Generate token
        token = jwt.encode({"sub": db_user.username}, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "username": db_user.username
        }
    except Exception as e:
        print(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post('/admin/login', response_model=Token)
async def admin_login(user: UserLogin):
    """Admin login with admin privileges"""
    try:
        # Find user
        db_user = await User.filter(username=user.username).first()
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid admin credentials"
            )
        
        # Verify password
        if not db_user.verify_password(user.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid admin credentials"
            )
        
        # Check if user is an admin
        if not db_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not an admin"
            )
        
        # Update last login time
        db_user.last_login = datetime.now()
        await db_user.save()
        
        # Generate admin token with admin claim
        token = jwt.encode(
            {"sub": db_user.username, "is_admin": True}, 
            settings.JWT_SECRET, 
            algorithm=settings.JWT_ALGORITHM
        )
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "username": db_user.username
        }
    except Exception as e:
        print(f"Admin login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get('/me', response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@router.get('/admin/users', response_model=List[UserResponse])
async def list_users(current_admin: User = Depends(get_current_admin_user)):
    """Admin endpoint to list all users"""
    users = await User.all()
    return users

@router.post('/admin/users/{user_id}/make-admin')
async def make_user_admin(user_id: int, current_admin: User = Depends(get_current_admin_user)):
    """Admin endpoint to make a user admin"""
    user = await User.filter(id=user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_admin = True
    await user.save()
    
    return {"message": f"User {user.username} is now an admin"}

@router.delete('/admin/users/{user_id}')
async def delete_user(user_id: int, current_admin: User = Depends(get_current_admin_user)):
    """Admin endpoint to delete a user"""
    user = await User.filter(id=user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    await user.delete()
    return {"message": f"User {user.username} deleted successfully"}

@router.post('/logout')
async def logout():
    """Logout endpoint (client should discard token)"""
    return {"message": "Successfully logged out"}
