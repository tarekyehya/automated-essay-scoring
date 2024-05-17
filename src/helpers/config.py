from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    DF_PATH: str
    TRAIN_PATH: str
    TEST_PATH: str
    TEST_SIZE: int
    class Config:
        env_file = ".env"


def get_settings():
    return Settings()