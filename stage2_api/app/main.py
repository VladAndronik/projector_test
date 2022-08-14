from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from api import api_router


app = FastAPI(title='Projector stage 2')


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    result: float


root_router = APIRouter()


@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)


app.include_router(api_router)
app.include_router(root_router)

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
