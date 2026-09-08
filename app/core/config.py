from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки сервиса. Читаются из переменных окружения и из .env файла"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Task API"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    debug: bool = False


settings = Settings()