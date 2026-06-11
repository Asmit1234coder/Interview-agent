import os
from dotenv import load_dotenv
import pyaudio

load_dotenv()

GEMINI_API_KEY:str=os.getenv("GEMINI_API_KEY","")

MODEL: str="gemini-3.1-flash-live-preview"

VOICE: str="Zephyr"
AUDIO_FORMAT: int=pyaudio.paInt16
CHANNELS: int =1
SEND_SAMPLE_RATE: int=16_000
RECEIVE_SAMPLE_RATE:int=24_000
CHUNK_SIZE:int=1_024

#Live API accepts at most 1 video frame per sec
FRAME_INTERNAL_SECONDS:float=1.0
MAX_FRAME_SIZE:tuple[int,int]=(768,768)
JPEG_QUALITY:int =85

SYSTEM_PROMPT: str= """you are a sharp,friendly assistant with live access to the user's camera.
    you can see what they show you in real life.Answer out loud,keep responses 
    short and conversational,and when user shows you something,describe or reason 
    about what you see.
    """


