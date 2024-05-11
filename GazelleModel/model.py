import torch
import transformers
from transformers import BitsAndBytesConfig
from .gazelle import (
    GazelleConfig,
    GazelleForConditionalGeneration,
)
from typing import Dict, List
from asyncio import Queue
import asyncio
from .log import configure_logger
from logging import Logger
from uuid import uuid4
import time
import numpy as np

logger:Logger = configure_logger(__name__)

class Model:
    def __init__(self,model_id:str):

        # models class id
        self.model_id = model_id
        
        # models names to loaded
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
        logger.info(f"[{self.model_id}] start loaded model in the model")
        self.__load_model()
        logger.info(f"[{self.model_id}] model is loaded with memory footprint of :-")
        logger.info(f"[{self.model_id}] GPU memory allocated: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

        # Model dtyepe config
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
            self.llm_config = GazelleConfig.from_pretrained(self.llm_model_id)
            logger.info(f"\t[{self.model_id}] loaded model config")
            self.llm_tokenizer = transformers.AutoTokenizer.from_pretrained(self.llm_model_id)
            logger.info(f"\t[{self.model_id}] loaded model tokenizer")
            self.audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
                self.audio_process_model_id
            )
            logger.info(f"\t[{self.model_id}] loaded model audio process")
            
            
            self.llm_model = GazelleForConditionalGeneration.from_pretrained(
                self.llm_model_id,
                device_map=self.device,
                quantization_config=self.quantization_config_8bit,
            )
            logger.info(f"\t[{self.model_id}] loaded LLM model")
    def bytes_to_float_array(self,audio_bytes):
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0
    
    async def putAudio(self, audio:bytes, prompt:str, request_id:str):
        logger.info(f"[{self.model_id}] got an audio with request id {request_id} and prompt {prompt}")
        
        audio = self.bytes_to_float_array(audio)
        __audio_values = self.audio_processor(
            audio=audio, 
            return_tensors="pt", 
            sampling_rate=self.samplerate).input_values
    
        __msgs = [
            {"role": "user", "content": prompt},
        ]
        try:
            __labels = self.llm_tokenizer.apply_chat_template(
                __msgs, return_tensors="pt", add_generation_prompt=True
            )
        except Exception as e:
            logger.error(f"error in labelmaker {e}")
        __payload = {
                "audio_values": __audio_values.squeeze(0).to(self.device).to(self.audio_dtype), 
                "input_ids": __labels.to(self.device), 
                "request_id": request_id
            } 
        logger.info(f"[{self.model_id}] llm payload is ready {__payload}")
        await self.audio_input_queue.put(__payload)


    async def __startLLM(self):
        self.start_llm = True
        logger.info(f"[{self.model_id}] llm model started acceptig the data")
        while self.start_llm:
            try:
                if not self.audio_input_queue.empty():
                    logger.info(f"[{self.model_id}] got something")
                    __payload:dict = await self.audio_input_queue.get()
                    
                    __request_id:str = __payload.pop("request_id")
                    logger.info(f"[{self.model_id}] llm model get request {__payload}")
                    try:
                        __llm_raw_token = self.llm_model.generate(__payload, max_new_tokens=64)
                        logger.info(f"RAW llm tokens {__llm_raw_token}")
                        __llm_output:str = self.llm_tokenizer.decode(__llm_raw_token[0])
                    except Exception as e:
                        logger.error(f"{e} | so put the basic value")
                        __llm_output = "Hello"
                    logger.info(f"[{self.model_id}] llm model output {__llm_output}")
                    await self.llm_output_queue.put({"text":__llm_output,"request_id":__request_id})
                else:
                    await asyncio.sleep(0.5)
            except Exception as e:
                pass
    def start(self): 
        self.llm_task = asyncio.create_task(self.__startLLM())
        asyncio.ensure_future(self.llm_task)
        logger.info(f"[{self.model_id}] llm task started")
        
    def stop(self):
        self.start_llm = False
        self.llm_task.cancel()
        logger.info(f"[{self.model_id}] llm task stoped")
        
    
    def ReturnTask(self):
        return self.llm_task
    


class LLMHandler:
    def __init__(self):
        self.request_handler:Dict[str:Queue] = dict()
        self.__models_count = 1
        self.__request_count = 0
        logger.info(f"number of model's in memroy {self.__models_count}")
        self.__models:List[Model] = [Model(str(uuid4())) for _ in range(self.__models_count)]
        logger.info("model list init")
        self.__LLMHandleTask = None
        self.__started:bool  = False
    
    def __startModels(self):
        for model in self.__models:
            model.start()

    def __stopModels(self):
        for model in self.__models:
            model.stop()
        torch.cuda.empty_cache()

    def register_client(self,request_id:str):
        logger.info(f"client with id {request_id} is registered")
        self.request_handler[request_id] = Queue()

    def unregister_client(self,request_id:str):
        logger.info(f"client with id {request_id} is unregistered")
        self.request_handler.pop(request_id)
    
    def start(self):
        self.__startModels()
        logger.info(f"LLM models where started")
        self.__LLMHandleTask = asyncio.create_task(self.__start())
        asyncio.ensure_future(self.__LLMHandleTask)
        logger.info(f"LLM Handler where started")

    def stop(self):
        self.__stopModels()
        logger.info(f"LLM models where stoped")
        self.__started = False
        self.__LLMHandleTask.cancel()
        logger.info(f"LLM Handler where stoped")

    async def putLLMRequest(self, audio:bytes, prompt:str, request_id:str):
        if request_id in self.request_handler:
            self.__request_count += 1
            model_no = self.__request_count % self.__models_count
            logger.info(f"client {request_id} with prompt {prompt} requested model id {self.__models[model_no].model_id} for audio chunck processing")
            try:
                await self.__models[model_no].putAudio(audio,prompt,request_id)
            except Exception as e:
                Logger.info(f"{e}")
            return True
        return False
    
    async def __start(self):
        logger.info(f"LLM handler process statred")
        self.__started = True
        while self.__started:
            await asyncio.sleep(0.5)
            for model in self.__models:
                if not model.llm_output_queue.empty():
                    payload = await model.llm_output_queue.get()
                    logger.info(f"client id {payload['request_id']} get llm output {payload['text']}")
                    await self.request_handler[payload['request_id']].put(payload['text'])
                    logger.info(f"request id put in queue {payload['text']}")
                    
    
    async def getLLMOutput(self,request_id:str):
        logger.info(f"output request for this client {request_id}")
        if not self.request_handler[request_id].empty():
            text = await self.request_handler[request_id].get()
            logger.info(f"llm out from queue")
            return text
        return False
    

    


            