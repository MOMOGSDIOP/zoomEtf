from pydantic import BaseSettings


class Settings(BaseSettings):
    YAHOO_API_KEY: str
    REDIS_URL: str = "redis://localhost:6379"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
