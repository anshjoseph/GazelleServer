from slm import Gazelle
from asyncio.queues import Queue
import numpy as np
import pyaudio
import asyncio




input_queue:Queue = Queue()
output_queue:Queue = Queue()
client = Gazelle(input_queue,output_queue)

# Recording Config
py_audio_stream = pyaudio.PyAudio()
chunk = 16000
format = pyaudio.paInt16
channels = 1
rate = 16000
record_seconds = 60000

async def recording():
    stream = py_audio_stream.open(
            format=format,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
    )
    for _ in range(0, int(rate / chunk * record_seconds)):
        await asyncio.sleep(0.3)
        try:
            data:bytes = stream.read(chunk, exception_on_overflow=False)
            await input_queue.put(data)
            print("print date in input queue")
        except Exception as e:
            print(e)

async def main():
    await client.connect()
    await asyncio.gather(client.start(),recording())

asyncio.run(main())