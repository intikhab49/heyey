import asyncio
from tortoise import Tortoise
from models.user import User, pwd_context
import asyncio
import os
from datetime import datetime
from tortoise import Tortoise, run_async
from models.user import User, pwd_context
from simple_config import settings

async def create_admin_user(username, password, email):
    # Connect to the database
    DATABASE_URL = os.getenv("DATABASE_URL", settings.DATABASE_URL)

    # Initialize Tortoise ORM
    await Tortoise.init(
        db_url=DATABASE_URL,
        modules={"models": ["models.user"]}
    )

    # Generate schema (this will create all tables)
    await Tortoise.generate_schemas(safe=True)

    try:
        # Check if user already exists
        existing_user = await User.filter(username=username).first()
        if existing_user:
            print(f"User '{username}' already exists.")
            if existing_user.is_admin:
                print(f"User '{username}' is already an admin.")
            else:
                existing_user.is_admin = True
                await existing_user.save()
                print(f"User '{username}' has been made an admin.")
            return

        # Create new admin user
        hashed_password = pwd_context.hash(password)
        now = datetime.now()

        await User.create(
            username=username,
            password_hash=hashed_password,
            email=email,
            is_admin=True,
            is_active=True,
            last_login=now,
            trades=0,
            balance='0 BTC'
        )
        print(f"Admin user '{username}' created successfully.")

    except Exception as e:
        print(f"Error creating admin user: {str(e)}")
        raise
    finally:
        await Tortoise.close_connections()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python create_admin.py <username> <password> [email]")
        sys.exit(1)

    username = sys.argv[1]
    password = sys.argv[2]
    email = sys.argv[3] if len(sys.argv) > 3 else f"{username}@example.com"

    asyncio.run(create_admin_user(username, password, email))
import os

async def create_admin_user(username, password):
    # Connect to the database
    DATABASE_URL = os.getenv("DATABASE_URL", settings.DATABASE_URL)
    
    # If using PostgreSQL, convert the URL format
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    await Tortoise.init(
        db_url=DATABASE_URL,
        modules={"models": ["models.user"]}
    )
    
    # Create schema if it doesn't exist
    await Tortoise.generate_schemas()
    
    # Check if admin already exists
    existing_admin = await User.filter(username=username).first()
    
    if existing_admin:
        if existing_admin.is_admin:
            print(f"Admin user '{username}' already exists.")
            return
        else:
            # Update existing user to be an admin
            existing_admin.is_admin = True
            await existing_admin.save()
            print(f"User '{username}' has been upgraded to admin.")
            return
    
    # Create new admin user
    hashed_password = pwd_context.hash(password)
    await User.create(
        username=username,
        password_hash=hashed_password,
        is_admin=True
    )
    
    print(f"Admin user '{username}' created successfully.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python create_admin.py <username> <password>")
        sys.exit(1)
    
    username = sys.argv[1]
    password = sys.argv[2]
    
    asyncio.run(create_admin_user(username, password))
    
    # Close connections
    asyncio.run(Tortoise.close_connections())
