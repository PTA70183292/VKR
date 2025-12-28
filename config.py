from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    database_url: str = "sqlite:///./sentiment.db"
    base_model_name: str = "DeepPavlov/rubert-base-cased-conversational"
    model_name: str = "DeepPavlov/rubert-base-cased-conversational"
    adapter_name: str = "egdfgdfgdsfg/rubert-base-cased-conversational"
    
    app_title: str = "Sentiment Analysis API"
    app_version: str = "1.0.0"
    
    class Config:
        env_file = None

settings = Settings()
