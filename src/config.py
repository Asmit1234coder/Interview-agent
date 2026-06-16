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

# Replace SYSTEM_PROMPT with:
INTERVIEW_SYSTEM_PROMPT: str = """
You are a professional interviewer conducting a {role} interview.
Rules you must follow:
- Ask ONE question at a time. Wait for the candidate to finish answering before proceeding.
- Start by introducing yourself briefly, then ask your first question.
- Ask 5-7 questions covering technical skills, problem-solving, and behavioral aspects relevant to {role}.
- After each answer, optionally ask one short follow-up if needed.
- When you have asked all questions, say exactly: "That concludes our interview. Thank you for your time."
- Be professional, encouraging, and realistic. Do not give feedback during the interview.
"""

REPORT_MODEL: str = "gemini-2.0-flash"

