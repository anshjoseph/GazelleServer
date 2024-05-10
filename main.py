
from fastapi import FastAPI, WebSocket, APIRouter
from fastapi.websockets import WebSocketDisconnect
from contextlib import asynccontextmanager
from GazelleModel import LLMHandler, log, Client
from typing import List
from uuid import uuid4
import asyncio
from uvicorn import Config, Server

logger = log.configure_logger(__name__)
loop = asyncio.new_event_loop()
connection_manager = None
llm_handler:LLMHandler = None
# clients:List[Client] = list()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_handler
    global connection_manager
    # Load the ML model
    llm_handler = LLMHandler()
    # connection_manager = ConnectionManager(llm_handler)
    logger.info("llm handler object is created")
    llm_handler.start()
    logger.info("llm handler process started")
    yield 
    # Clean up the ML models and release the resources
    llm_handler.stop()
    logger.info("llm handler process stoped")





app = FastAPI(lifespan=lifespan)




ls = []
@app.websocket("/connection")
async def WebScoketConnectionHandler(websocket:WebSocket):
    await websocket.accept()
    client = Client(websocket,llm_handler,str(uuid4()))
    ls.append(client)
    await client.start()




if __name__ == "__main__":
    config = Config(app=app, loop=loop, reload=True, host="0.0.0.0", port=4000)
    server = Server(config)
    loop.run_until_complete(server.serve())
