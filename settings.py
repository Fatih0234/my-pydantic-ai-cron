from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parent

class Settings(BaseSettings):
    OPENCHARGEMAP_API_KEY: str
    MARKETSTACK_API_KEY: str
    OPENAI_API_KEY: str

    STORAGE_FILE_PATH: Path = ROOT_DIR / "data.csv"

    # For Gmail API: typically you'd have either OAuth2 tokens, or a service account, etc.
    # This example is a placeholder for your actual method:
    GMAIL_CREDENTIALS_FILE: str = "credentials.json"  # or any other approach

    # If using a token file for OAuth
    GMAIL_TOKEN_FILE: str = "token.json"

    # Gmail settings
    GMAIL_USER: str = Field("", description="Gmail address for sending notifications")
    GMAIL_APP_PASSWORD: str = Field("", description="Gmail App Password for authentication")

    class Config:
        # If you prefer .env files:
        env_file = ".env"
        env_file_encoding = "utf-8"

# A single global settings instance:
settings = Settings()
