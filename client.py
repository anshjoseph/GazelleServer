import numpy as np
import pyaudio
import asyncio
import websocket
import time
import base64
import websockets.client
import websockets.connection
from GazelleModel.log import configure_logger
from GazelleModel import write_bytesIO
import json
import torch
import torchaudio
import librosa
import io
from scipy.io.wavfile import write



# sound,sr =librosa.load("./samples/test16.wav",sr=16000)
# with open("/media/ansh/22A048A8A04883EF/WORK/BOLNA/GazelleServer/samples/test21.wav",'rb') as file:
#     sound = file.read()

logger = configure_logger(__name__)

py_audio_stream = pyaudio.PyAudio()

ws_url = "ws://44.221.66.152:4000/connection"
def bytes_to_float_array(audio_bytes):
    raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
    return raw_data.astype(np.float32) / 32768.0
connection = websocket.create_connection(ws_url)


chunk = 16000*3
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
                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                # audio_array = np.array(bytes_to_float_array(data),dtype=np.float32)
                    __data = np.frombuffer(data, dtype='int16')
                    print(f"len of it {len(__data)}")
                    audio_bytes = write_bytesIO(16000,__data)
                    with open("temp.wav",'wb') as file:
                        file.write(audio_bytes.read())
                    with open("/media/ansh/22A048A8A04883EF/WORK/BOLNA/GazelleServer/temp.wav",'rb') as file:
                        __data = file.read()
                    file = io.BytesIO(__data)
                    # file.write(audio_bytes.read())
                    # audio, sr = torchaudio.load(file)
                    # print(sr)
                    connection.send_bytes(__data)
                except Exception as e:
                    print("error")
                    print(e)
                
                
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