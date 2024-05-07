
from fastapi import FastAPI, WebSocket, APIRouter
import uvicorn
from contextlib import asynccontextmanager
from GazelleModel import Client, LLMHandler
from typing import List


llm_handler:LLMHandler = None
clients:List[Client] = list()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    llm_handler = LLMHandler()
    llm_handler.start()
    yield
    # Clean up the ML models and release the resources
    llm_handler.stop()





app = FastAPI(lifespan=lifespan)



@app.websocket("/connection")
async def WebScoketConnectionHandler(websocket:WebSocket):
    await websocket.accept()
    client = Client(websocket,llm_handler)
    clients.append(client)



if __name__ == "__main__":
    uvicorn.run("main:app",host='127.0.0.1',port=8000,reload=True)