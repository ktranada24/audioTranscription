import queue
import threading
import time
import numpy as np
import torch
from features.asr_features import build_mel_filterbank, waveform_to_log_mel
from inference.decode import load_model, transcribe_features
from features.audio_processing import stream_48k_file_to_pipeline



pipeline_queue = queue.Queue(maxsize=100)

model = load_model("checkpoints/best_val.pt")

mel_filterbank = build_mel_filterbank(
    sample_rate=16000,
    n_fft_bins=201,
    n_mels=80
)

audio_thread = threading.Thread(
    target=stream_48k_file_to_pipeline,
    args=(input_file, pipeline_queue),
    daemon=True
)    

while True:
    
    clean_chunk = pipeline_queue.get()

    features = waveform_to_log_mel(clean_chunk, mel_filterbank)

    pred = transcribe_features(features, model)

    feature_buffer.append(features)

    full_features = torch.cat(feature_buffer, dim=0 )

    if full_features.shape[0] >= DECODE_WINDOW_FRAMES:
    
        pred = transcribe_features( full_features, model)
        
        print("partial transcript:", pred)
        
        feature_buffer = []
        
    pipeline_queue.task_done()
    
audio_thread.join()
    
print("Done.")    