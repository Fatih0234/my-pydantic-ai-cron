from pathlib import Path

from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parent

class Settings(BaseSettings):
    OPENCHARGEMAP_API_KEY: str
    MARKETSTACK_API_KEY: str
    OPENAI_API_KEY: str

    STORAGE_FILE_PATH: Path = ROOT_DIR / "data.csv"

    class Config:
        # If you prefer .env files:
        env_file = ".env"
        env_file_encoding = "utf-8"

# A single global settings instance:
settings = Settings()
