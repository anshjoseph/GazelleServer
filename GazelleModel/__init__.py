from .model import LLMHandler
from fastapi import WebSocket
from .log import configure_logger
from uuid import uuid4
import asyncio




logger = configure_logger(__name__)
class Client:
    def __init__(self,ws:WebSocket,llm_handler:LLMHandler) -> None:
        self.ws:WebSocket = ws
        self.client_id:str = str(uuid4())
        self.llm_handler:LLMHandler = llm_handler
        self.llm_handler.register_client(self.client_id)


        self.recevier_task = None
        self.sender_task = None
    async def recevier(self):
        while True:
            revc:bytes = await self.ws.receive_bytes()

            if revc == b"END":
                logger.info(f"[{self.client_id}] conversation end")
                pass
            elif revc == b"START":
                logger.info(f"[{self.client_id}] conversation stated")
                pass
            else:
                self.llm_handler.putLLMRequest(revc,"assist me",self.client_id)
    async def sender(self):
        while True:
            out = self.llm_handler.getLLMOutput(self.client_id)
            if out:
                await self.ws.send(out)
    def start(self):
        self.recevier_task = asyncio.create_task(self.recevier())
        self.sender_task = asyncio.create_task(self.sender())
        logger.info(f"[{self.client_id}] client sender and receiver task started")
    
    def stop(self):
        self.recevier_task.cancel()
        self.sender_task.cancel()
        logger.info(f"[{self.client_id}] client sender and receiver task stoped")
