from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Database settings
    db_host: str = "localhost"
    db_user: str = "root"
    db_password: str = ""
    db_name: str = "dev_drinkindia_db"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Fuzzy matching threshold
    match_threshold: int = 80
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
