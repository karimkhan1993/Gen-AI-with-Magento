import uvicorn
from fastapi import FastAPI
from api.routes import router
from utils.logging_config import setup_logging
from config import get_settings

app = FastAPI(title="Magento Gen AI API")
app.include_router(router)

if __name__ == "__main__":
    settings = get_settings()
    setup_logging()
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)