
from fastapi import FastAPI, WebSocket, APIRouter
import uvicorn
import json
from GazelleModel import Client, LLMHandler
from typing import List


app = FastAPI()
llm_handler:LLMHandler = None
clients:List[Client] = list()


@app.on_event("startup")
def startup():
    global llm_handler
    llm_handler = LLMHandler()
    llm_handler.start()
@app.on_event("shutdown")
def shutdown():
    global llm_handler
    llm_handler.stop()

@app.websocket("/connection")
async def WebScoketConnectionHandler(websocket:WebSocket):
    await websocket.accept()
    client = Client(websocket,llm_handler)
    clients.append(client)



if __name__ == "__main__":
    uvicorn.run("main:app",host='127.0.0.1',port=8000,reload=True)