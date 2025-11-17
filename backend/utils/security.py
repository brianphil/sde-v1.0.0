"""Security utilities for authentication and authorization.

Provides:
- Password hashing and verification (bcrypt)
- JWT token generation and validation
- Token payload extraction
- Password strength validation
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ===========================
# Password Functions
# ===========================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password string
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password to verify against

    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def validate_password_strength(password: str) -> tuple[bool, Optional[str]]:
    """Validate password meets minimum requirements.

    Requirements:
    - At least 8 characters
    - Contains at least one letter
    - Contains at least one number

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter"

    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"

    return True, None


# ===========================
# JWT Token Functions
# ===========================

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token.

    Args:
        data: Payload data to encode in token (should include 'sub' for user ID)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT refresh token.

    Args:
        data: Payload data to encode in token (should include 'sub' for user ID)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
    """Verify and decode JWT token.

    Args:
        token: JWT token string
        token_type: Expected token type ('access' or 'refresh')

    Returns:
        Decoded token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        # Verify token type
        if payload.get("type") != token_type:
            logger.warning(f"Token type mismatch: expected {token_type}, got {payload.get('type')}")
            return None

        # Check expiration (automatically handled by jwt.decode, but we can add custom logic)
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            logger.debug("Token has expired")
            return None

        return payload

    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error verifying token: {e}")
        return None


def get_token_subject(token: str) -> Optional[str]:
    """Extract subject (user ID) from token.

    Args:
        token: JWT token string

    Returns:
        User ID (subject) if valid, None otherwise
    """
    payload = verify_token(token)
    if payload:
        return payload.get("sub")
    return None


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode token without verification (for debugging).

    WARNING: This does NOT verify the token signature.
    Use verify_token() for production code.

    Args:
        token: JWT token string

    Returns:
        Decoded payload or None
    """
    try:
        return jwt.decode(token, options={"verify_signature": False})
    except Exception as e:
        logger.error(f"Failed to decode token: {e}")
        return None


# ===========================
# Token Refresh
# ===========================

def refresh_access_token(refresh_token: str) -> Optional[str]:
    """Generate new access token from refresh token.

    Args:
        refresh_token: Valid refresh token

    Returns:
        New access token if refresh token is valid, None otherwise
    """
    payload = verify_token(refresh_token, token_type="refresh")

    if not payload:
        return None

    # Extract user info from refresh token
    user_id = payload.get("sub")
    username = payload.get("username")
    role = payload.get("role")

    if not user_id:
        logger.warning("Refresh token missing 'sub' claim")
        return None

    # Create new access token with same user info
    new_token_data = {
        "sub": user_id,
        "username": username,
        "role": role,
    }

    return create_access_token(new_token_data)


# ===========================
# Validation Helpers
# ===========================

def validate_username(username: str) -> tuple[bool, Optional[str]]:
    """Validate username format.

    Requirements:
    - 3-50 characters
    - Alphanumeric and underscores only
    - Must start with letter

    Args:
        username: Username to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"

    if len(username) > 50:
        return False, "Username must be at most 50 characters long"

    if not username[0].isalpha():
        return False, "Username must start with a letter"

    if not all(c.isalnum() or c == '_' for c in username):
        return False, "Username can only contain letters, numbers, and underscores"

    return True, None


def validate_email(email: str) -> tuple[bool, Optional[str]]:
    """Validate email format (basic check).

    Args:
        email: Email address to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not email or '@' not in email:
        return False, "Invalid email format"

    if len(email) > 255:
        return False, "Email must be at most 255 characters long"

    # Basic validation
    parts = email.split('@')
    if len(parts) != 2:
        return False, "Invalid email format"

    local, domain = parts
    if not local or not domain:
        return False, "Invalid email format"

    if '.' not in domain:
        return False, "Invalid email domain"

    return True, None
