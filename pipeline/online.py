import queue
import threading
import time
import numpy as np
import torch
from features.asr_features import build_mel_filterbank, waveform_to_log_mel
from inference.decode import load_model, transcribe_features
from features.audio_processing import stream_48k_file_to_pipeline
from postprocess.onlineprocess import postprocess_online

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 0.100
DECODE_WINDOW_SECONDS = 1.0
FRAMES_PER_SECOND = 100  # because hop = 10 ms
DECODE_WINDOW_FRAMES = int(DECODE_WINDOW_SECONDS * FRAMES_PER_SECOND)

model = load_model("checkpoints/best_val.pt")


def run_online():

    pipeline_queue = queue.Queue(maxsize=100)
    
    mel_filterbank = build_mel_filterbank(
        sample_rate=16000,
        n_fft_bins=201,
        n_mels=80
    )

    audio_buffer = np.array([], dtype=np.float32)

    DECODE_WINDOW_SAMPLES = 16000  # 1 second
    DECODE_STRIDE_SAMPLES = 4800   # 300 ms

    audio_thread = threading.Thread(
        target=stream_48k_file_to_pipeline,
        args=(input_file, pipeline_queue),
        daemon=True
    )

    audio_thread.start()

    while True:

        clean_chunk = pipeline_queue.get()

        if clean_chunk is None:
            pipeline_queue.task_done()
            break

        audio_buffer = np.concatenate([audio_buffer, clean_chunk])

        while len(audio_buffer) >= DECODE_WINDOW_SAMPLES:

            window = audio_buffer[:DECODE_WINDOW_SAMPLES]

            features = waveform_to_log_mel(window, mel_filterbank)
            pred = transcribe_features(features, model)
            pred = postprocess_online(pred)

            print("partial transcript:", pred)

            # Slide forward, but keep overlap/context

            audio_buffer = audio_buffer[DECODE_STRIDE_SAMPLES:]

        pipeline_queue.task_done()

    audio_thread.join()

    print("Done.")