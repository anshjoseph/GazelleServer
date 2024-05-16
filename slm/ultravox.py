from asyncio import Queue
from .base import SLM
import asyncio
import base64
from openai import OpenAI
import os

"""
import base64
from openai import OpenAI

client = OpenAI(base_url="https://ultravox.api.fixie.ai/v1", api_key="89d313170fb06cc6b9e5933c29ea9353")

with open('./male1.wav', 'rb') as infile:
    audio_data = infile.read()
    base64_audio = base64.b64encode(audio_data).decode('utf-8')


completion = client.chat.completions.create(
    model= "fixie-ai/ultravox-v0.1",
    messages=[{
        'role': 'user',
        'content': [
            {
                "type": "text",
                "text": "What's in the <|audio|>?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:audio/wav;base64,{base64_audio}"
                }
            }
        ]
    }]
)

print(completion.choices[0].message)
"""



class Ultravox(SLM):
    def __init__(self, input_queue: Queue, output_queue: Queue, agent_name:str = "gazelle_slm", prompt:str="some thing") -> None:
        super().__init__(input_queue, output_queue)
        # class parameter
        print(os.getenv('ultravox_api'))
        self.agent_name:str = agent_name
        # model connection
        self.client:OpenAI = OpenAI(base_url=os.getenv('ultravox_api'), api_key=os.getenv('ultravox'))

        # task for event loop
        self.llm_task  = None
        # model prompt
        self.prompt:str = prompt
    
    async def __startLLM(self):
        print("SENDER task on")
        async for audio_frame, duration, is_frame_ready in self.make_frames():
            await asyncio.sleep(0.1)
            if is_frame_ready:
                audio_bytes_to_send:bytes = await self.pred_frame(audio_frame)
                base64_audio = base64.b64encode(audio_bytes_to_send).decode('utf-8')
                completion = self.client.chat.completions.create(
                        model= "fixie-ai/ultravox-v0.1",
                        messages=[{
                            'role': 'user',
                            'content': [
                                {
                                    "type": "text",
                                    "text": "What's in the <|audio|>?"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:audio/wav;base64,{base64_audio}"
                                    }
                                }
                            ]
                        }]
                    )
                output = completion.choices[0].message
                await self.output_queue.put(output)
                print(completion.choices[0].message)
            else:
                continue

    
    
    
    async def start(self):
        await super().start()
        self.llm_task = asyncio.create_task(self.__startLLM())
        await asyncio.gather(self.llm_task)

    async def stop(self):
        self.llm_task.cancel()
    