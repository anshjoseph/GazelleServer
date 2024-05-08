
from fastapi import FastAPI, WebSocket, APIRouter
import uvicorn
from contextlib import asynccontextmanager
from GazelleModel import Client, LLMHandler, log
from typing import List

logger = log.configure_logger(__name__)

llm_handler:LLMHandler = None
clients:List[Client] = list()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    llm_handler = LLMHandler()
    logger.info("llm handler object is created")
    llm_handler.start()
    logger.info("llm handler process started")
    yield
    # Clean up the ML models and release the resources
    llm_handler.stop()
    logger.info("llm handler process stoped")





app = FastAPI(lifespan=lifespan)



@app.websocket("/connection")
async def WebScoketConnectionHandler(websocket:WebSocket):
    await websocket.accept()
    client = Client(websocket,llm_handler)
    logger.info(f"new client with id {client.client_id} is joined")
    clients.append(client)



if __name__ == "__main__":
    uvicorn.run("main:app",host='127.0.0.1',port=8000,reload=True)