import asyncio
from google import genai
from google.genai import types
from google.genai.live import AsyncSession

from audio import AudioIo
from webcam import Camera
from config import (
    CHUNK_SIZE,
    FRAME_INTERNAL_SECONDS,
    GEMINI_API_KEY,
    MODEL,
    SEND_SAMPLE_RATE,
    SYSTEM_PROMPT,
    VOICE,
)

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=SYSTEM_PROMPT,
    speech_config=types.SpeechConfig(   # ✅ fix typo: was speecConfig
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoice(voice_name=VOICE)
        )
    ),
    output_audio_transcription=types.AudioTranscriptionConfig(
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=25_000,
            sliding_window=types.SlidingWindow(target_tokens=12_800),
        ),
    ),
)


class LiveVisionAgent:
    """Four asyncio tasks sharing one live API session"""

    def __init__(self) -> None:
        """Set up API client, devices, and playback queue"""
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._audio = AudioIo()
        self._camera = Camera()
        self._playback_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._session: AsyncSession | None = None

    async def _stream_microphone(self) -> None:
        """Read raw PCM chunks from mic and send them upstream"""
        mic = self._audio.open_mic()
        mime_type = f"audio/pcm:rate={SEND_SAMPLE_RATE}"
        while True:
            chunk = await asyncio.to_thread(
                mic.read, CHUNK_SIZE, exception_on_overflow=False
            )
            await self._session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=mime_type)
            )

    async def _stream_camera(self) -> None:
        """Send one JPEG frame per second (LIVE API maximum)"""
        while True:
            frame = await asyncio.to_thread(self._camera.read_jpeg_frame)
            if frame is not None:
                await self._session.send_realtime_input(
                    video=types.Blob(data=frame, mime_type="image/jpeg")
                )
            await asyncio.sleep(FRAME_INTERNAL_SECONDS)

    async def _receive_responses(self) -> None:
        """Queue model's audio for playback and print its transcript"""
        while True:
            async for message in self._session.receive():
                content = message.server_content
                if content is None:
                    continue
                if content.interrupted:
                    await self._drain_playback_queue()
                if content.model_turn:
                    for part in content.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            await self._playback_queue.put(part.inline_data.data)
                if content.output_transcription:
                    print("Model transcript:", content.output_transcription.text)

    async def _playback_audio(self) -> None:
        """Play queued audio chunks through speaker"""
        speaker = self._audio.open_speaker()
        while True:
            chunk = await self._playback_queue.get()
            speaker.write(chunk)

    async def _drain_playback_queue(self) -> None:
        """Clear any unplayed audio when interrupted"""
        while not self._playback_queue.empty():
            self._playback_queue.get_nowait()

    async def run(self) -> None:
        """Start live session and run all tasks"""
        async with self._client.live.connect(model=MODEL, config=LIVE_CONFIG) as session:
            self._session = session
            await asyncio.gather(
                self._stream_microphone(),
                self._stream_camera(),
                self._receive_responses(),
                self._playback_audio(),
            )
