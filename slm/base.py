from faster_whisper.vad import SileroVADModel
from .utils import write_bytesIO
import io
from asyncio.queues import Queue
import aiofiles
import numpy as np


class SLM:
    def __init__(self,input_queue:Queue,output_queue:Queue) -> None:
        # IO interface
        self.input_queue:Queue = input_queue
        self.output_queue:Queue = output_queue
        # VAD config
        self.vad_threshold:float = 0.5
        self.VAD:SileroVADModel = SileroVADModel("./VAD/silero_vad.onnx")
        self.vad_state = self.VAD.get_initial_state(batch_size=1)
        # Frame config
        self.slience_count:int = 0
        self.slience_count_threshold_split_frame:int = 2
        # interruption handler
        self.interruption_signal:bool = False 
        # Audio config
        self.sample_rate:int = 16000
        self.temp_file_loc:str = "temp.wav"
        # modules status
        self.status:str = "INIT" 
    
    @staticmethod
    def int_to_float_array(audio_array)->np.array:
        "needed to convert for VAD return(NDarray)"
        return audio_array.astype(np.float32) / 32768.0

    async def make_frames(self):
        """
        yield (frame:NDarray, duration:float, is_frame_ready:bool)
        """
        print("START making frame")
        frame:list = []
        duration:float = 0.00
        while True:
            try:
                audio_chunk:bytes = await self.input_queue.get()
                audio_array = np.frombuffer(audio_chunk,dtype=np.int16)
                __audio_duration = len(audio_array) / self.sample_rate
                prob,self.vad_state = self.VAD(SLM.int_to_float_array(audio_array),self.vad_state,self.sample_rate)
                duration += __audio_duration
                print(f"FRAME STATUS: {duration} {prob} {self.slience_count}")
                if prob[0][0] < self.vad_threshold:
                    self.slience_count += 1
                    # no one is speecking
                    self.interruption_signal = False
                else:
                    frame.extend(audio_array)
                    self.slience_count = 0
                    # some one is speecking
                    self.interruption_signal = True
            except Exception as e:
                print(e)
            if self.slience_count >= self.slience_count_threshold_split_frame:
                if len(frame) != 0:
                    yield (np.asarray(frame,dtype=np.int16),duration,True)
                    # clean up stuff
                    frame.clear()
                    duration = 0.00
                    self.slience_count = 0
            # yield (np.zeros(1),duration,False)
    async def pred_frame(self,frame:np.array)->bytes:
        audio_bytes:io.BytesIO = write_bytesIO(self.sample_rate,frame)
        # we have optimized it
        ret:bytes = b""
        async with aiofiles.open("temp.wav",'wb') as file:
            print("FILE SAVE IN TEMP")
            await file.write(audio_bytes.read())
        async with aiofiles.open("temp.wav",'rb') as file:
            ret = await file.read()
        return ret

    # abstract functions
    async def start(self):
        self.status = "START"
    async def connect(self):
        self.status = "CONNECTED"
    async def disconect(self):
        self.status = "DISCONNECTED"

    async def InterruptionSignal(self):
        return self.interruption_signal
    
