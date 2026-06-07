# auth/security.py — API-key auth + rate limiting.
import os

from fastapi import Header, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])


def require_api_key(x_api_key: str = Header(default=None)):
    """Auth is disabled when no API_KEYS are configured (dev-friendly default)."""
    if not API_KEYS:
        return
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
