from .model import LLMHandler
from fastapi import WebSocket, WebSocketDisconnect
from .log import configure_logger
import asyncio
import json




logger = configure_logger(__name__)
class Client:
    def __init__(self,websocket:WebSocket, llm_handler:LLMHandler, client_id:str):
        self.websocket:WebSocket = websocket
        self.llm_handler:LLMHandler = llm_handler
        self.client_id:str = client_id
        self.tasks = []
        self.llm_handler.register_client(self.client_id)
        self.__stop:bool = False
    async def sender(self):
        logger.info(f"sender is started")
        while True:
            try:
                await asyncio.sleep(0.5)
                
                llm_output = await self.llm_handler.getLLMOutput(self.client_id)
                try:
                    await self.websocket.send_text(llm_output)
                except TypeError:
                    pass
            except Exception as e:
                break
    async def recever(self):
        logger.info(f"recever is started")
        while True:
            try:
                await asyncio.sleep(0.5)
                audio_json:bytes = json.loads(await self.websocket.receive_text())
                await self.llm_handler.putLLMRequest(audio_json["audio_chunk"],audio_json["prompt"],self.client_id)
            except Exception as e:
                break

    async def start(self):
        self.__stop = False
        self.tasks.append(asyncio.create_task(self.sender()))
        self.tasks.append(asyncio.create_task(self.recever()))
        await asyncio.gather(*self.tasks)
    def stop(self):
        self.__stop = True
        if self.__stop:
            self.llm_handler.unregister_client(self.client_id)
