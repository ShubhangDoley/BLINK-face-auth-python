"""
BLINK — PostgreSQL Connection Diagnostic Utility
Checks if the connection string in .env is configured correctly
and can successfully reach the database server.
"""
import os
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARN] python-dotenv is not installed. Reading from system env only.")

from sqlalchemy import create_engine

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    print("[ERROR] DATABASE_URL is not set in environment or .env file.")
    sys.exit(1)

print(f"[DIAG] Attempting connection to: {db_url.split('@')[-1] if '@' in db_url else db_url} (credentials hidden)")

try:
    # Set a short timeout (5s) for diagnostic speed
    engine = create_engine(db_url, connect_args={"connect_timeout": 5} if db_url.startswith("postgresql") else {})
    with engine.connect() as conn:
        print("[SUCCESS] Connection established! PostgreSQL database is reachable.")
        
        # Test table verification
        from models import Base
        print("[DIAG] Synchronizing database tables...")
        Base.metadata.create_all(bind=engine)
        print("[SUCCESS] Tables synchronized/created successfully.")
        
except Exception as e:
    print(f"\n[FAILED] Connection error: {e}")
    print("\nPlease verify:")
    print(" 1. The PostgreSQL server is running.")
    print(" 2. Host, port, username, and password are correct in .env.")
    print(" 3. The database name exists on the PostgreSQL server.")
    sys.exit(1)
