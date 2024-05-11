import numpy as np
import pyaudio
import asyncio
import websocket
import time
import base64
import websockets.client
import websockets.connection
from GazelleModel.log import configure_logger
import json
import torch
import torchaudio
import librosa



# sound,sr =librosa.load("./samples/test16.wav",sr=16000)
with open("./samples/test16.wav",'rb') as file:
    sound = file.read()

logger = configure_logger(__name__)

py_audio_stream = pyaudio.PyAudio()

ws_url = "ws://44.221.66.152:4000/connection"
def bytes_to_float_array(audio_bytes):
    raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
    return raw_data.astype(np.float32) / 32768.0
connection = websocket.create_connection(ws_url)


chunk = 12000
format = pyaudio.paInt16
channels = 1
rate = 16000
record_seconds = 60000

async def recordAudio():
    stream = py_audio_stream.open(
            format=format,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
    )
    print("audio listing ...")
    print(int(rate / chunk * record_seconds))
    while True:
        try:
            for _ in range(0, int(rate / chunk * record_seconds)):
                await asyncio.sleep(0.5)
                data = stream.read(chunk, exception_on_overflow=False)
                # audio_array = np.array(bytes_to_float_array(data),dtype=np.float32)
                connection.send_bytes(sound)
                
        except Exception as e:
            print(e)


    
async def recevingOutput():
    print("start receving")
    while True:
        try:
            await asyncio.sleep(0.5)
            data = connection.recv()
            print(data)
        except Exception as e:
            print(e)
            break

async def main():
    tasks = []
    
        
    tasks.append(asyncio.create_task(recordAudio()))
    tasks.append(asyncio.create_task(recevingOutput()))
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"STOP {e}")

asyncio.run(main())