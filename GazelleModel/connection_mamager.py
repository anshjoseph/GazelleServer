from .client import Client
from .model import LLMHandler
from .log import configure_logger
import asyncio
from uuid import uuid4
from fastapi import WebSocket, WebSocketDisconnect, WebSocketException
from typing import Dict



logger = configure_logger(__name__)

class ConnectionManager:
    def __init__(self, llm_handler:LLMHandler) -> None:
        self.websockets_connection:Dict[str,Client] = dict()
        self.llm_handler:LLMHandler = llm_handler
    
    async def datafeeder(self):
        while True:
            pass

    async def connection(self, ws:WebSocket):
        client_id:str = str(uuid4())
        await ws.accept()
        self.websockets_connection[ws] = None
        # self.websockets_connection[ws] = Client(ws, self.llm_handler, client_id)
        # self.websockets_connection[ws].start()
    
    async def disconnect(self, ws:WebSocket):
        # self.websockets_connection[ws].stop()
        del self.websockets_connection[ws]


