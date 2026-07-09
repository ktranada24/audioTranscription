import time
import wave
import numpy as np
import scipy.signal as signal
from pyrnnoise import RNNoise
import scipy.io.wavfile as wavfile
import threading
import queue

# Initialize the denoiser once, globally
denoiser = RNNoise(sample_rate=48000)

def rnnoise_denoise_48k_chunk(chunk_48k_int16):
    """
    CORE FUNCTION: RNNoise only. No VAD dropping. No downsampling.
    Returns: (clean_48k_float, avg_speech_prob)
    """
    clean_frames = []
    speech_probabilities = []
    
    for speech_prob, denoised_frame in denoiser.denoise_chunk(chunk_48k_int16):
        clean_frames.append(denoised_frame.flatten()) 
        speech_probabilities.append(np.mean(speech_prob))
        
    if not clean_frames:
        return np.array([], dtype=np.float32), 0.0
        
    avg_speech_prob = sum(speech_probabilities) / len(speech_probabilities)
    clean_48k_float = np.concatenate(clean_frames)
    
    return clean_48k_float.astype(np.float32), avg_speech_prob

def _apply_dsp_filters_16k(clean_48k_float): # downsamples audio from 48kHz to 16kHz bc ASR engines usually dislike 48kHz sample rate
    """
    HELPER FUNCTION: Keeps our math DRY. Applies downsampling, pre-emphasis, and DC removal.
    """
    clean_16k_float = signal.resample_poly(clean_48k_float, up=1, down=3)
    clean_16k_float = np.append(clean_16k_float[0], clean_16k_float[1:] - 0.97 * clean_16k_float[:-1]) # pre emphasis filter
    clean_16k_float = clean_16k_float - np.mean(clean_16k_float) # DC offset removal.
    return clean_16k_float.astype(np.float32)

def kyle_online_preprocess_chunk(chunk_48k_int16, vad_threshold=0.2, chunk_duration_ms=100): # denoises with rnnoise
    """
    LIVE APP FUNCTION: Calls RNNoise core. 
    Returns exactly the dictionary format Nikhil requested.
    """
    clean_48k_float, avg_speech_prob = rnnoise_denoise_48k_chunk(chunk_48k_int16)
    
    # If silence or empty, return the silence heartbeat dictionary
    if avg_speech_prob < vad_threshold or len(clean_48k_float) == 0:
        return {"type": "silence", "duration_ms": chunk_duration_ms}
        
    # If speech, apply DSP and return the speech dictionary
    clean_16k_chunk = _apply_dsp_filters_16k(clean_48k_float)
    return {"type": "speech", "audio": clean_16k_chunk}

def kyle_training_preprocess_chunk(chunk_48k_int16):
    """
    TRAINING FUNCTION: Calls RNNoise core. NEVER drops audio. Downsamples everything.
    """
    clean_48k_float, _ = rnnoise_denoise_48k_chunk(chunk_48k_int16)
    
    if len(clean_48k_float) == 0:
        return np.array([], dtype=np.float32)
        
    return _apply_dsp_filters_16k(clean_48k_float)

def stream_48k_file_to_pipeline(file_path, pipeline_queue, chunk_size=4800): # chunk size will be 1600 samples later after downsampling 3x
    try:
        with wave.open(file_path, 'rb') as wf:
            assert wf.getframerate() == 48000, "Test file must be 48kHz!"
            assert wf.getnchannels() == 1, "Test file must be Mono, not Stereo!"
            
            chunk_duration = chunk_size / 48000.0 

            while True:
                raw_bytes = wf.readframes(chunk_size)
                
                if not raw_bytes:
                    break
                    
                if len(raw_bytes) < chunk_size * 2: # if nearing eof & remaining samples < 4800, pad with 00 binary data.
                    raw_bytes = raw_bytes.ljust(chunk_size * 2, b'\x00')
                    
                chunk_48k_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                
                # --- Function handles the dictionary generation ---
                payload = kyle_online_preprocess_chunk(
                    chunk_48k_int16, 
                    vad_threshold=0.3, 
                    chunk_duration_ms=100
                )
                pipeline_queue.put(payload, block=True)
                # --------------------------------------------------------
                
                time.sleep(chunk_duration)
    except Exception as e:
        print(f"Background thread crashed: {e}")
    finally:
        # THE NEW POISON PILL
        pipeline_queue.put({"type": "end"})

if __name__ == "__main__": # main thread
    print("Starting Audio Processing Test")
    
    test_queue = queue.Queue(maxsize=100) # the conveyer belt where main waits for audio to come in
    input_file = "uhh.wav"
    output_file = "clean_output_16k.wav"
    
    stream_thread = threading.Thread(
        target=stream_48k_file_to_pipeline, 
        args=(input_file, test_queue)
    )
    stream_thread.start()
    
    collected_chunks = []
    print("Listening to queue and collecting clean audio...")
    
    while True:
        payload = test_queue.get(timeout=5.0)
        
        # Check for Nikhil's new "end" poison pill format
        if payload["type"] == "end":
            print("End signal received. File streaming is completely finished.")
            break
            
        # Receiver logic based on Nikhil's exact keys
        if payload["type"] == "speech":
            collected_chunks.append(payload["audio"]) # main thread grabs NumPy array of downsampled, cleaned speech from dict

        elif payload["type"] == "silence":
            pass
            
        test_queue.task_done()
            
    print("File streaming finished. Stitching audio back together")
    
    if collected_chunks:
        full_audio_float32 = np.concatenate(collected_chunks)
        max_amp = np.max(np.abs(full_audio_float32))
        print(f"Diagnostic: Max audio amplitude is {max_amp:.2f}")
        
        if max_amp > 2.0: # checks if max amplitude in whole audio is too loud, clips if yes
            safe_audio = np.clip(full_audio_float32, -32768.0, 32767.0)
        else:
            if max_amp > 0: # if audio is normal, normalize whole audio file ... (0 is silence, +/- 32767 is ceiling & floor respectively)
                safe_audio = (full_audio_float32 / max_amp) * 0.9 * 32767.0
            else:
                safe_audio = full_audio_float32 * 32767.0
                
        full_audio_int16 = safe_audio.astype(np.int16)
        wavfile.write(output_file, 16000, full_audio_int16)
        print(f"Success. isolated vocals here: {output_file}")
    else:
        print("No audio was collected.")