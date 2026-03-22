from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configuración de la aplicación"""

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="app.log", alias="LOG_FILE")

    # Snowflake
    snowflake_account: str = Field(alias="SNOWFLAKE_ACCOUNT")
    snowflake_user: str = Field(alias="SNOWFLAKE_USER")
    snowflake_password: str = Field(alias="SNOWFLAKE_PASSWORD")
    snowflake_warehouse: str = Field(alias="SNOWFLAKE_WAREHOUSE")
    snowflake_database: str = Field(alias="SNOWFLAKE_DATABASE")
    snowflake_schema: str = Field(alias="SNOWFLAKE_SCHEMA")

    class Config:
        env_file = ".env"
        extra = "ignore"