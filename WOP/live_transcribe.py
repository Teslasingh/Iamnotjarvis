import os
import re
import time
import queue
import threading

import numpy as np
import sounddevice as sd
import ollama
from faster_whisper import BatchedInferencePipeline, WhisperModel
from piper.voice import PiperVoice

# ================= SETTINGS =================

WHISPER_MODEL = "large-v3"
# Local folder where the Whisper weights are cached after first download.
# Once populated, the model loads straight from disk with no network
# round-trip to Hugging Face on startup.
WHISPER_MODEL_DIR = os.path.join("models", f"whisper-{WHISPER_MODEL}")
OLLAMA_MODEL = "llama3.1:8b"

DEVICE = "cuda"
COMPUTE = "int8_float16"

SAMPLE_RATE = 16000

BLOCK_DURATION = 0.1
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

START_THRESHOLD = 0.1
# Time of silence before we consider the utterance finished. Kept fairly
# tight since vad_filter=True already cleans up the transcription input,
# so we don't need extra silence margin for accuracy's sake.
SILENCE_SECONDS = 1

# How long to keep ignoring the mic after TTS finishes speaking, to let
# any echo/room-reflection tail decay before we start listening again.
ECHO_COOLDOWN_SECONDS = 0.35

PIPER_VOICE_MODEL = "models/en_GB-alba-medium.onnx"

SYSTEM_PROMPT = """
You are a helpful AI voice assistant.
Rules:
- Give direct answers.
- Try to keep responses short.
- Be conversational.
- No Markdown formatting in your responses.
"""

# Sentence-boundary regex: split after . ! ? that are followed by
# whitespace or end of string (keeps decimals like "3.14" intact-ish
# for short conversational replies; good enough for TTS chunking).
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Clause/sentence-boundary regex used ONLY to slice off the very first
# chunk sent to TTS (commas, semicolons, colons, or full sentence ends),
# so a long opening sentence doesn't delay time-to-first-audio. Later
# chunks use full-sentence splitting for natural prosody.
FIRST_CHUNK_BOUNDARY_RE = re.compile(r"(?<=[,;:.!?])\s+")

# ============================================


class TextToSpeech:
    """Wraps Piper TTS with a persistent output stream and a background
    worker thread so speech playback never blocks LLM generation or
    microphone capture."""

    def __init__(self, model_path: str):
        self.voice = None
        self.stream = None
        self.queue: "queue.Queue[str | None]" = queue.Queue()
        self.synth_fn = None

        # Tracks whether audio is queued or actively being played, so the
        # mic loop can be gated off for the whole duration (not just
        # while one sentence is playing) — avoids the assistant hearing
        # and transcribing its own voice.
        self._lock = threading.Lock()
        self._pending = 0

        if not os.path.exists(model_path):
            print(f"⚠️  Piper model not found at {model_path}")
            return

        try:
            self.voice = PiperVoice.load(model_path)
            self.stream = sd.RawOutputStream(
                samplerate=self.voice.config.sample_rate,
                channels=1,
                dtype="int16",
            )
            self.stream.start()

            # Resolve the synthesis method ONCE instead of hasattr-checking
            # on every call.
            if hasattr(self.voice, "synthesize_stream_raw"):
                self.synth_fn = self._synth_stream_raw
            else:
                self.synth_fn = self._synth_fallback

            threading.Thread(target=self._run, daemon=True).start()

            # Warm up synthesis so the first *real* response isn't slowed
            # down by cold-start init (CUDA/ONNX session setup etc).
            for _ in self.synth_fn("Hi"):
                pass

            print("✅ Piper TTS loaded successfully with: en_GB-alba-medium")
        except Exception as e:
            print(f"❌ Error loading Piper TTS: {e}")
            self.voice = None

    # ---- synthesis backends ----

    def _synth_stream_raw(self, text: str):
        for audio_bytes in self.voice.synthesize_stream_raw(text):
            yield np.frombuffer(audio_bytes, dtype=np.int16)

    def _synth_fallback(self, text: str):
        audio = self.voice.synthesize(text)
        if isinstance(audio, (bytes, bytearray)):
            yield np.frombuffer(audio, dtype=np.int16)
            return

        import io
        import wave

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)
        wav_buffer.seek(0)
        wav_file = wave.open(wav_buffer, "rb")
        audio_data = wav_file.readframes(wav_file.getnframes())
        yield np.frombuffer(audio_data, dtype=np.int16)

    # ---- worker loop ----

    def _run(self):
        while True:
            text = self.queue.get()
            if text is None:  # shutdown signal
                break
            try:
                for chunk in self.synth_fn(text):
                    self.stream.write(chunk)
            except Exception as e:
                print(f"\nTTS playback error: {e}")
            finally:
                with self._lock:
                    self._pending -= 1

    # ---- public API ----

    def enqueue(self, text: str):
        """Queue a sentence/chunk of text to be spoken. Non-blocking."""
        if self.voice and text.strip():
            with self._lock:
                self._pending += 1
            self.queue.put(text)

    def is_speaking(self) -> bool:
        """True if audio is queued and/or currently playing."""
        with self._lock:
            return self._pending > 0


def get_whisper_model_path() -> str:
    """Return a local path to the Whisper weights, downloading them once
    if they aren't already cached. On every run after the first, this
    skips the network entirely and loads straight from disk."""
    marker = os.path.join(WHISPER_MODEL_DIR, "model.bin")

    if os.path.exists(marker):
        print(f"Loading Whisper from local cache ({WHISPER_MODEL_DIR})...")
        return WHISPER_MODEL_DIR

    print(f"Downloading Whisper ({WHISPER_MODEL}) — first run only...")
    from faster_whisper.utils import download_model

    os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)
    download_model(WHISPER_MODEL, output_dir=WHISPER_MODEL_DIR)
    return WHISPER_MODEL_DIR


whisper_path = get_whisper_model_path()
whisper = WhisperModel(
    whisper_path,
    device=DEVICE,
    compute_type=COMPUTE,
    local_files_only=True,
)
pipeline = BatchedInferencePipeline(model=whisper)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("Loading Ollama...")
ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "Hi"}])

print("Loading Piper TTS...")
tts = TextToSpeech(PIPER_VOICE_MODEL)

print("Ready!\n")

audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()


def callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())


def ask_llm(prompt: str):
    """Stream the LLM response, printing tokens live and dispatching each
    completed chunk to TTS as soon as it's ready. The very first chunk is
    sliced on clause boundaries (commas, semicolons) instead of waiting
    for a full sentence, to minimize time-to-first-audio on long opening
    sentences. Every chunk after that uses full-sentence splitting for
    natural prosody."""
    print("\n🤖 Assistant: ", end="", flush=True)

    messages.append({"role": "user", "content": prompt})

    response = ""
    pending = ""  # text not yet flushed to TTS
    first_chunk_sent = False
    MIN_FIRST_CHUNK_CHARS = 12

    stream = ollama.chat(
        model=OLLAMA_MODEL,
        stream=True,
        messages=messages,
        options={
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_predict": 256,
            "num_ctx": 4096,
        },
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        response += token
        pending += token

        if not first_chunk_sent:
            boundary = None
            for m in FIRST_CHUNK_BOUNDARY_RE.finditer(pending):
                if m.start() >= MIN_FIRST_CHUNK_CHARS:
                    boundary = m
                    break
            if boundary:
                tts.enqueue(pending[: boundary.start() + 1])
                pending = pending[boundary.end():]
                first_chunk_sent = True
            continue

        # After the first chunk, split on full sentence boundaries only.
        parts = SENTENCE_SPLIT_RE.split(pending)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                tts.enqueue(sentence)
            pending = parts[-1]

    # Flush whatever's left (final sentence, or short replies with no
    # terminal punctuation).
    if pending.strip():
        tts.enqueue(pending)

    print("\n")

    messages.append({"role": "assistant", "content": response})
    return response


with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=BLOCK_SIZE,
    callback=callback,
):
    speaking = False
    silence_time = 0.0
    audio_buffer = []
    cooldown_until = 0.0  # monotonic timestamp; ignore mic until past this

    try:
        while True:
            block = audio_queue.get()

            # --- Full-duplex gate: while the assistant is talking (or a
            # sentence is still queued to be spoken), don't look at the
            # mic at all. Just discard the block — no buffering, no
            # partial transcription state carried across the gap.
            if tts.is_speaking():
                cooldown_until = time.monotonic() + ECHO_COOLDOWN_SECONDS
                if speaking:
                    # We were mid-utterance when TTS kicked in (shouldn't
                    # normally happen, but reset defensively) — drop it
                    # rather than transcribing a mixed/interrupted clip.
                    speaking = False
                    silence_time = 0.0
                    audio_buffer = []
                continue

            # Echo tail cooldown after TTS just stopped.
            if time.monotonic() < cooldown_until:
                continue

            volume = np.sqrt(np.mean(block ** 2))

            if volume > START_THRESHOLD:
                if not speaking:
                    speaking = True
                    silence_time = 0
                    audio_buffer = []
                    print("\n🎤 Listening...")

                audio_buffer.append(block)
                silence_time = 0

            elif speaking:
                audio_buffer.append(block)
                silence_time += BLOCK_DURATION

                if silence_time >= SILENCE_SECONDS:
                    print("🧠 Transcribing...")

                    audio = np.concatenate(audio_buffer).flatten()

                    segments, _ = pipeline.transcribe(
                        audio,
                        batch_size=8,
                        beam_size=1,
                        language="en",
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=300,
                            speech_pad_ms=200,
                        ),
                        condition_on_previous_text=False,
                    )

                    text = "".join(segment.text for segment in segments).strip()

                    if text:
                        print(f"\n👤 You: {text}")
                        ask_llm(text)

                    speaking = False
                    silence_time = 0
                    audio_buffer = []

    except KeyboardInterrupt:
        print("\nStopped.")
        tts.queue.put(None)