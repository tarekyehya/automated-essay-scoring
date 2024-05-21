from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    RAW_PATH: str
    DF_PATH: str
    TRAIN_PATH: str
    TEST_PATH: str
    TEST_SIZE: int
    QWK_A: float
    QWK_B: float
    N_SPLITS: int
    MODELS_CLASSIC_PATH: str
    class Config:
        env_file = ".env"


def get_settings():
    return Settings()