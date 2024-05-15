from asyncio import Queue
from .base import SLM
import websocket
import asyncio
import pickle

class Gazelle(SLM):
    def __init__(self, input_queue: Queue, output_queue: Queue, agent_name:str = "gazelle_slm", agent_access_ws:str = "ws://44.221.66.152:4000/connection",prompt:str="assist me") -> None:
        super().__init__(input_queue, output_queue)
        # for debug stuff
        self.__debug = False
        # class parameter
        self.agent_name:str = agent_name
        self.agent_access_ws:str = agent_access_ws
        # ws connection
        self.ws_connection = None
        # task for event loop
        self.sender_task  = None
        self.recever_task = None
        # model prompt
        self.prompt:str = prompt
    
    async def __sender(self):
        print("SENDER task on")
        async for audio_frame, duration, is_frame_ready in self.make_frames():
            await asyncio.sleep(0.5)
            if is_frame_ready:
                try:
                    audio_bytes_to_send:bytes = await self.pred_frame(audio_frame)
                except Exception as e:
                    print(f"__sender {e}")
                if not self.__debug:
                    self.ws_connection.send_bytes(pickle.dumps({"audio_chunk":audio_bytes_to_send ,"prompt":f"{self.prompt}<|audio|>"}))
            else:
                continue
    async def __recever(self):
        print("RECEVER task on")
        while True:
            try:
                await asyncio.sleep(0.5)
                if not self.__debug:
                    data = self.ws_connection.recv()
                else:
                    data = "some thing"
                await self.output_queue.put(data)
            except Exception as e:
                print(e)
                break

    async def connect(self):
        await super().connect()
        if not self.__debug:
            self.ws_connection = websocket.create_connection(self.agent_access_ws)
        else:
            self.ws_connection = None
    
    async def disconect(self):
        await super().disconect()
        if not self.__debug:
            self.ws_connection.close()
        else:
            self.ws_connection = None
    
    async def start(self):
        await super().start()
        self.sender_task = asyncio.create_task(self.__sender())
        self.recever_task = asyncio.create_task(self.__recever())
        await asyncio.gather(*[self.sender_task,self.recever_task])

    async def stop(self):
        self.sender_task.cancel()
        self.recever_task.cancel()
    