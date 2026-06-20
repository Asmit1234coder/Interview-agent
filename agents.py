"""
agents.py – Interview Agent
Streams mic + webcam to Gemini Live API, acts as a professional interviewer,
accumulates a full transcript log, and generates a structured post-interview report.
"""

import asyncio
import json
import re
from datetime import datetime

from google import genai
from google.genai import types
from google.genai.live import AsyncSession

from audio import AudioIo
from webcam import Camera
from src.config import (
    CHUNK_SIZE,
    FRAME_INTERNAL_SECONDS,
    GEMINI_API_KEY,
    MODEL,
    REPORT_MODEL,
    INTERVIEW_SYSTEM_PROMPT,
    VOICE,
    SEND_SAMPLE_RATE,
)

# ── Phrase Gemini must say to signal the interview is over ────────────────────
END_PHRASE = "that concludes our interview"


def _build_live_config(role: str) -> types.LiveConnectConfig:
    """Build a fresh LiveConnectConfig with the role injected into the system prompt."""
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=INTERVIEW_SYSTEM_PROMPT.format(role=role),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
            )
        ),
        # Transcribe BOTH sides of the conversation
        output_audio_transcription=types.AudioTranscriptionConfig(),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=25_000,
            sliding_window=types.SlidingWindow(target_tokens=12_800),
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Live Interview Agent
# ══════════════════════════════════════════════════════════════════════════════

class InterviewAgent:
    """
    Four asyncio tasks sharing one Gemini Live session.
    Accumulates a structured transcript log.
    Sets self.interview_ended = True when Gemini says the closing phrase.
    """

    def __init__(self, role: str = "Software Engineer") -> None:
        self._role = role
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._audio = AudioIo()
        self._camera = Camera()
        self._playback_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._session: AsyncSession | None = None

        # Public state read by the UI
        self.transcript_log: list[dict] = []   # {"speaker", "text", "ts"}
        self.interview_ended: bool = False
        self.error: str | None = None

        # Optional callback so UI can receive entries in real time
        self._transcript_push_callback = None

    def set_transcript_callback(self, cb):
        """UI sets this to receive transcript entries as they arrive."""
        self._transcript_push_callback = cb

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _push_entry(self, speaker: str, text: str):
        text = text.strip()
        if not text:
            return
        entry = {
            "speaker": speaker,
            "text": text,
            "ts": datetime.now().strftime("%H:%M:%S"),
        }
        self.transcript_log.append(entry)
        if self._transcript_push_callback:
            self._transcript_push_callback(entry)

        # Detect natural interview end from Gemini's closing phrase
        if speaker == "interviewer" and END_PHRASE in text.lower():
            self.interview_ended = True

    # ── Async streaming tasks ─────────────────────────────────────────────────

    async def _stream_microphone(self) -> None:
        """Read raw PCM chunks from mic and send them upstream."""
        mic = self._audio.open_mic()
        mime_type = f"audio/pcm;rate={SEND_SAMPLE_RATE}"
        while True:
            chunk = await asyncio.to_thread(
                mic.read, CHUNK_SIZE, exception_on_overflow=False
            )
            await self._session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=mime_type)
            )

    async def _stream_camera(self) -> None:
        """Send one JPEG frame per second to Gemini."""
        while True:
            frame = await asyncio.to_thread(self._camera.read_jpeg_frame)
            if frame is not None:
                await self._session.send_realtime_input(
                    video=types.Blob(data=frame, mime_type="image/jpeg")
                )
            await asyncio.sleep(FRAME_INTERNAL_SECONDS)

    async def _receive_responses(self) -> None:
        """Queue audio for playback and push both sides of transcript."""
        while True:
            async for message in self._session.receive():
                content = message.server_content
                if content is None:
                    continue

                if content.interrupted:
                    await self._drain_playback_queue()

                # Queue Gemini's audio chunks for speaker playback
                if content.model_turn:
                    for part in content.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            await self._playback_queue.put(part.inline_data.data)

                # Interviewer (model output) transcript
                if content.output_transcription and content.output_transcription.text:
                    self._push_entry("interviewer", content.output_transcription.text)

                # Candidate (mic input) transcript
                if content.input_transcription and content.input_transcription.text:
                    self._push_entry("candidate", content.input_transcription.text)

    async def _playback_audio(self) -> None:
        """Write queued PCM audio to speaker stream."""
        speaker = self._audio.open_speaker()
        while True:
            chunk = await self._playback_queue.get()
            speaker.write(chunk)

    async def _drain_playback_queue(self) -> None:
        """Clear unplayed audio chunks when Gemini is interrupted."""
        while not self._playback_queue.empty():
            self._playback_queue.get_nowait()

    # ── Main entry point ──────────────────────────────────────────────────────

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        """
        Connect to Gemini Live and run all four tasks concurrently.
        Stops when stop_event is set OR interview_ended becomes True.
        """
        config = _build_live_config(self._role)
        try:
            async with self._client.aio.live.connect(
                model=MODEL, config=config
            ) as session:
                self._session = session

                tasks = [
                    asyncio.create_task(self._stream_microphone()),
                    asyncio.create_task(self._stream_camera()),
                    asyncio.create_task(self._receive_responses()),
                    asyncio.create_task(self._playback_audio()),
                ]

                # Poll until stopped or interview naturally ends
                while True:
                    await asyncio.sleep(0.4)
                    if stop_event and stop_event.is_set():
                        break
                    if self.interview_ended:
                        # Give Gemini 2 s to finish its closing sentence
                        await asyncio.sleep(2.0)
                        break

                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            self.error = str(e)
            raise
        finally:
            self._audio.close()
            self._camera.close()
            await self._client.aio.aclose()


# ══════════════════════════════════════════════════════════════════════════════
# Report Generator  (uses standard Gemini API, not Live)
# ══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """
    Takes the structured transcript log from InterviewAgent and calls the
    standard Gemini API to produce a JSON evaluation report.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    def generate(self, role: str, transcript_log: list[dict]) -> dict:
        """
        Returns a dict with keys:
          overall_score, summary, scores (dict), strengths (list),
          weaknesses (list), verdict, recommendation
        """
        if not transcript_log:
            return self._empty_report()

        formatted = "\n".join(
            f"[{e['ts']}] {e['speaker'].upper()}: {e['text']}"
            for e in transcript_log
        )

        prompt = f"""You are a senior hiring manager. Evaluate this {role} interview transcript and return a structured JSON report.

TRANSCRIPT:
{formatted}

Return ONLY a valid JSON object with exactly these keys — no markdown, no explanation, no code fences:
{{
  "overall_score": <integer 1-10>,
  "summary": "<2-3 sentence overview of the candidate's overall performance>",
  "scores": {{
    "technical_knowledge": <integer 1-10>,
    "communication": <integer 1-10>,
    "problem_solving": <integer 1-10>,
    "confidence": <integer 1-10>,
    "role_fit": <integer 1-10>
  }},
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "weaknesses": ["<area for improvement 1>", "<area for improvement 2>", "<area for improvement 3>"],
  "verdict": "<one of exactly: Strong Hire | Hire | Maybe | No Hire>",
  "recommendation": "<2 sentence hiring recommendation>"
}}"""

        response = self._client.models.generate_content(
            model=REPORT_MODEL,
            contents=prompt,
        )

        raw = response.text.strip()

        # Strip markdown fences if model ignores the instruction
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Report generation failed — model returned invalid JSON.\n"
                f"Error: {e}\nRaw output:\n{raw}"
            )

    @staticmethod
    def _empty_report() -> dict:
        return {
            "overall_score": 0,
            "summary": "No interview data was recorded.",
            "scores": {
                "technical_knowledge": 0,
                "communication": 0,
                "problem_solving": 0,
                "confidence": 0,
                "role_fit": 0,
            },
            "strengths": [],
            "weaknesses": [],
            "verdict": "No Hire",
            "recommendation": "Interview could not be evaluated — no transcript was captured.",
        }