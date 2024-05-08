import numpy as np
import pyaudio
import asyncio
import websocket
from websocket import WebSocket
import time

def bytes_to_float_array(audio_bytes):
    raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
    return raw_data.astype(np.float32) / 32768.0

ws_url = "ws://44.221.66.152:8000/connection"
py_audio_stream = pyaudio.PyAudio()

connection:WebSocket = websocket.create_connection(ws_url)


chunk = 8192
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
    try:
        for _ in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk, exception_on_overflow=False)
            audio_array = bytes_to_float_array(data)
            
            t1 = time.time()
            connection.send_bytes(audio_array.tobytes())
            print("out recevied at delta: "+time.time()-t1)
    except KeyboardInterrupt:
        print("error happend")
async def recevingOutput():
    while connection.connected:
        data = connection.recv()

