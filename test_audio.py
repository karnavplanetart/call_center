import collections
import sys
import signal
import time
import pyaudio
import webrtcvad
import os

# Signal handler for graceful termination
def signal_handler(sig, frame):
    print('Terminating...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class AudioStream:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=16000,
                                      input=True,
                                      frames_per_buffer=320)  # 320 bytes for 20ms frame at 16kHz

    def read(self):
        try:
            return self.stream.read(320, exception_on_overflow=False)
        except IOError as e:
            print(f"Error reading from stream: {e}")
            return b''

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, audio_stream):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    start_time = time.time()

    while True:
        frame = audio_stream.read()
        if len(frame) == 0:
            continue  # Skip empty frames resulting from overflow errors

        try:
            is_speech = vad.is_speech(frame, sample_rate)
        except webrtcvad.VadError as e:
            print(f"Error processing frame: {e}")
            continue

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend([f for f, s in ring_buffer])
                ring_buffer.clear()
                start_time = None  # Reset timer on speech detection
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
                start_time = time.time()  # Start timer on silence detection

        # Check for silence duration
        if not triggered and start_time:
            yield None  # Indicate silence

def main():
    vad = webrtcvad.Vad(1)
    audio_stream = AudioStream()
    sample_rate = 16000
    frame_duration_ms = 20
    padding_duration_ms = 300

    print("Listening for voice activity...")

    silence_start_time = None

    for audio_chunk in vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, audio_stream):
        if audio_chunk:
            silence_start_time = None
            print(f"[{time.strftime('%H:%M:%S')}] Voice activity detected.")
        else:
            if silence_start_time is None:
                silence_start_time = time.time()
            else:
                silence_duration = time.time() - silence_start_time
                print(f"[{time.strftime('%H:%M:%S')}] Silence duration: {silence_duration:.2f} seconds")
                if 4 < silence_duration <= 9:
                    print("Hello, are you still there?")
                    os.system('say "Hello, are you still there?"')
                elif silence_duration > 17:
                    print("Cancelling call, no voice activity.")
                    os.system('say "Cancelling call, no voice activity."')
                    break

    audio_stream.close()

if __name__ == "__main__":
    main()