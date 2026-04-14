import secrets
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from app.config.config import get_settings

# Set auto_error=False to allow us to handle the error manually based on the flag
security = HTTPBasic(auto_error=False)


def get_current_username(credentials: Optional[HTTPBasicCredentials] = Depends(security)):
    settings = get_settings()

    # Bypass authentication if disabled
    if not settings.IS_AUTH_ENABLED:
        return credentials.username if credentials else "anonymous"

    # If enabled and no credentials provided, raise error (since auto_error is False)
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = settings.API_USERNAME.encode("utf8")
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )

    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = settings.API_PASSWORD.encode("utf8")
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
