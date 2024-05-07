import torch
import transformers
from transformers import BitsAndBytesConfig
from gazelle import (
    GazelleConfig,
    GazelleForConditionalGeneration,
)
from typing import Dict, List
from queue import Queue
import asyncio



class Model:
    def __init__(self):
        self.llm_model_id = "tincans-ai/gazelle-v0.2"
        self.audio_process_model_id = "facebook/wav2vec2-base-960h"
        self.is_model_loaded = False
        # MODEL
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_config = None
        self.audio_processor = None
        self.quantization_config_8bit = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        # basic conf
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__load_model()
        self.audio_dtype = torch.float16
        self.samplerate:int = 16000
        self.start_llm:bool = False

        # TASK
        self.llm_task = None

        #  QUEUE
        self.audio_input_queue:Queue = Queue()
        self.llm_output_queue:Queue = Queue()
    def __load_model(self):
        if not self.is_model_loaded:
            self.llm_config = GazelleConfig.from_pretrained(self.model_id)
            self.llm_tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            self.audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            self.llm_model = GazelleForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map=self.device,
                quantization_config=self.quantization_config_8bit,
            )
    def putAudio(self,audio:bytes, prompt:str, request_id:str):
        __audio_values = self.audio_processor(audio=audio, return_tensors="pt", sampling_rate=self.samplerate).input_values
        __msgs = [
            {"role": "user", "content": prompt},
        ]
        __labels = self.tokenizer.apply_chat_template(
            __msgs, return_tensors="pt", add_generation_prompt=True
        )
        __payload = {"audio_values": __audio_values.squeeze(0).to(self.device).to(self.audio_dtype), "input_ids": __labels.to(self.device), "request_id": request_id} 
        self.audio_input_queue.put_nowait(__payload)

        self.llm_output_queue.empty()

    async def __startLLM(self):
        self.start_llm = True
        while self.start_llm:
            try:
                __payload:dict = self.audio_input_queue.get(timeout=2)
                __request_id:str = __payload.pop("request_id")
                __llm_output:str = self.llm_tokenizer.decode(self.llm_model.generate(__payload, max_new_tokens=64)[0])
                self.llm_output_queue.put_nowait({"text":__llm_output,"request_id":__request_id})
            except Exception as e:
                pass
    def start(self):
        self.llm_task = asyncio.create_task(self.__startLLM())
        
    def stop(self):
        self.start_llm = False
        self.llm_task.cancel()
    


class LLMHandler:
    def __init__(self):
        self.request_handler:Dict[str:Queue] = dict()
        self.__models_count = 1
        self.__request_count = 0
        self.__models:List[Model] = [Model() for _ in range(self.__models_count)]
        self.__LLMHandleTask = None
        self.__started:bool  = False
    
    def __startModels(self):
        for model in self.__models:
            model.start()

    def __stopModels(self):
        for model in self.__models:
            model.stop()

    def register_client(self,request_id:str):
        self.request_handler[request_id] = Queue()

    def unregister_client(self,request_id:str):
        del self.request_handler.pop(request_id)
    
    def start(self):
        self.__startModels()
        self.__LLMHandleTask = asyncio.create_task(self.__start())

    def stop(self):
        self.__stopModels()
        self.__started = False
        self.__LLMHandleTask.cancel()

    def putLLMRequest(self,audio:bytes,prompt:str,request_id:str):
        if request_id in self.request_handler:
            self.__request_count += 1
            model_no = self.__request_count % self.__models_count
            self.__models[model_no].putAudio(audio,prompt,request_id)
            return True
        return False
    
    async def __start(self):
        self.__started = True
        while self.__started:
            for model in self.self.__models:
                if not model.llm_output_queue.empty():
                    payload = model.llm_output_queue.get()
                    self.request_handler[payload['request_id']].put(payload['text'])
    
    def getLLMOutput(self,request_id:str):
        if len(self.request_handler[request_id]) > 0:
            return self.request_handler[request_id].pop(0)
        return False
    

    


            