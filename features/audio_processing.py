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

def _apply_dsp_filters_16k(clean_48k_float):
    """
    HELPER FUNCTION: Keeps our math DRY. Applies downsampling, pre-emphasis, and DC removal.
    """
    clean_16k_float = signal.resample_poly(clean_48k_float, up=1, down=3)
    clean_16k_float = np.append(clean_16k_float[0], clean_16k_float[1:] - 0.97 * clean_16k_float[:-1])
    clean_16k_float = clean_16k_float - np.mean(clean_16k_float)
    return clean_16k_float.astype(np.float32)

def kyle_online_preprocess_chunk(chunk_48k_int16, vad_threshold=0.2):
    """
    LIVE APP FUNCTION: Calls RNNoise core. Drops audio if silent. Downsamples if speech.
    """
    clean_48k_float, avg_speech_prob = rnnoise_denoise_48k_chunk(chunk_48k_int16)
    
    if avg_speech_prob < vad_threshold or len(clean_48k_float) == 0:
        return np.array([], dtype=np.float32)
        
    return _apply_dsp_filters_16k(clean_48k_float)

def kyle_training_preprocess_chunk(chunk_48k_int16):
    """
    TRAINING FUNCTION: Calls RNNoise core. NEVER drops audio. Downsamples everything.
    """
    clean_48k_float, _ = rnnoise_denoise_48k_chunk(chunk_48k_int16)
    
    if len(clean_48k_float) == 0:
        return np.array([], dtype=np.float32)
        
    return _apply_dsp_filters_16k(clean_48k_float)

def stream_48k_file_to_pipeline(file_path, pipeline_queue, chunk_size=4800):
    try:
        with wave.open(file_path, 'rb') as wf:
            assert wf.getframerate() == 48000, "Test file must be 48kHz!"
            assert wf.getnchannels() == 1, "Test file must be Mono, not Stereo!"
            
            chunk_duration = chunk_size / 48000.0 

            while True:
                raw_bytes = wf.readframes(chunk_size)
                
                if not raw_bytes:
                    break
                    
                if len(raw_bytes) < chunk_size * 2: 
                    raw_bytes = raw_bytes.ljust(chunk_size * 2, b'\x00')
                    
                chunk_48k_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                
                # --- USE YOUR NEW ONLINE FUNCTION ---
                clean_16k_chunk = kyle_online_preprocess_chunk(chunk_48k_int16, vad_threshold=0.3)
                
                if len(clean_16k_chunk) > 0:
                    pipeline_queue.put({
                        "status": "SPEECH", 
                        "audio": clean_16k_chunk
                    }, block=True)
                else:
                    pipeline_queue.put({
                        "status": "SILENCE", 
                        "duration": chunk_duration 
                    }, block=True)
                
                time.sleep(chunk_duration)
    except Exception as e:
        print(f"Background thread crashed: {e}")
    finally:
        pipeline_queue.put(None)


if __name__ == "__main__":
    print("Starting Audio Processing Test")
    
    test_queue = queue.Queue(maxsize=100)
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
        
        # Check for the poison pill
        if payload is None:
            print("Poison pill received. File streaming is completely finished.")
            break
            
        # --- NEW TEST RECEIVER LOGIC ---
        if payload["status"] == "SPEECH":
            collected_chunks.append(payload["audio"])
        elif payload["status"] == "SILENCE":
            # For our WAV file test, we just ignore silence and keep waiting
            pass
        # -------------------------------
            
        test_queue.task_done()
            
    print("File streaming finished. Stitching audio back together")
    
    if collected_chunks:
        full_audio_float32 = np.concatenate(collected_chunks)
        max_amp = np.max(np.abs(full_audio_float32))
        print(f"Diagnostic: Max audio amplitude is {max_amp:.2f}")
        
        if max_amp > 2.0:
            safe_audio = np.clip(full_audio_float32, -32768.0, 32767.0)
        else:
            if max_amp > 0:
                safe_audio = (full_audio_float32 / max_amp) * 0.9 * 32767.0
            else:
                safe_audio = full_audio_float32 * 32767.0
                
        full_audio_int16 = safe_audio.astype(np.int16)
        wavfile.write(output_file, 16000, full_audio_int16)
        print(f"Success. isolated vocals here: {output_file}")
    else:
        print("No audio was collected.")