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

def isolate_vocals_rnnoise_optimized(chunk_48k_int16, vad_threshold=0.6):
    """
    Takes a raw 48kHz int16 chunk, cleans it, downsamples to 16kHz float32.
    Only returns audio if human speech is detected (VAD).
    """
    clean_frames = []
    speech_probabilities = []
    
    # Process the chunk (RNNoise automatically slices it into 10ms frames)
    for speech_prob, denoised_frame in denoiser.denoise_chunk(chunk_48k_int16):
        
        # FLATTEN the (1, 480) matrix into a standard 1D array of 480 samples
        clean_frames.append(denoised_frame.flatten()) 
        
        # Use np.mean() just in case speech_prob also returned as an array!
        speech_probabilities.append(np.mean(speech_prob))
        
    if not clean_frames:
        return np.array([], dtype=np.float32)
        
    # --- VAD LOGIC ---
    # Calculate the average probability that someone is talking in this 100ms chunk
    avg_speech_prob = sum(speech_probabilities) / len(speech_probabilities)
    
    # If it falls below our threshold, consider it silence/background noise
    if avg_speech_prob < vad_threshold:
        return np.array([], dtype=np.float32) 
    # -----------------
        
    # Combine the processed 10ms frames back into a single continuous array
    clean_48k_float = np.concatenate(clean_frames)
    
    # Downsample directly to 16kHz for Nikhil (Exact 1:3 ratio)
    clean_16k_float = signal.resample_poly(clean_48k_float, up=1, down=3)
    clean_16k_float = np.append(clean_16k_float[0], clean_16k_float[1:] - 0.97 * clean_16k_float[:-1]) # Pre emphasis filter
    clean_16k_float = clean_16k_float - np.mean(clean_16k_float) # removes DC offset
    return clean_16k_float.astype(np.float32)

def stream_48k_file_to_pipeline(file_path, pipeline_queue, chunk_size=4800):
    try:
        with wave.open(file_path, 'rb') as wf:
            assert wf.getframerate() == 48000, "Test file must be 48kHz!"
            assert wf.getnchannels() == 1, "Test file must be Mono, not Stereo!"
            
            chunk_duration = chunk_size / 48000.0 

            while True:
                raw_bytes = wf.readframes(chunk_size)
                
                # If we've reached the end of the file, break the loop
                if not raw_bytes:
                    break
                    
                # If the very last chunk is smaller than 4800, pad it with zeros 
                # so RNNoise doesn't crash from a dimension mismatch
                if len(raw_bytes) < chunk_size * 2: # *2 because 16-bit is 2 bytes per sample
                    raw_bytes = raw_bytes.ljust(chunk_size * 2, b'\x00')
                    
                chunk_48k_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                
                clean_16k_chunk = isolate_vocals_rnnoise_optimized(chunk_48k_int16, vad_threshold=0.2)
                
                if len(clean_16k_chunk) > 0:
                    pipeline_queue.put(clean_16k_chunk, block=True)
                
                time.sleep(chunk_duration)
    except Exception as e:
        print(f"Background thread crashed: {e}")
    finally:
        # THE POISON PILL: Tell the main thread we are permanently done
        pipeline_queue.put(None)


if __name__ == "__main__":
    print("Starting Audio Processing Test")
    
    # 1. Create the shared memory queue
    test_queue = queue.Queue(maxsize=100)
    
    # Define your test file path (Make sure this file exists in your folder!)
    input_file = "uhh.wav"
    output_file = "clean_output_16k.wav"
    
    # 2. Start streaming the file in a background thread (Simulating Kyle's loop)
    stream_thread = threading.Thread(
        target=stream_48k_file_to_pipeline, 
        args=(input_file, test_queue)
    )
    stream_thread.start()
    
    # 3. Collect the output (Simulating Nikhil's loop)
    collected_chunks = []
    print("🎧 Listening to queue and collecting clean audio...")
    
    while True:
        # Wait up to 5 seconds for a chunk (generous buffer for heavy CPU processing)
        clean_chunk = test_queue.get(timeout=5.0)
        
        # Check for the poison pill
        if clean_chunk is None:
            print("Poison pill received. File streaming is completely finished.")
            break
            
        collected_chunks.append(clean_chunk)
        test_queue.task_done()
            
    print("File streaming finished. Stitching audio back together")
    
    # 4. Save the result safely (THIS IS THE BLOCK YOU WERE MISSING)
    if collected_chunks:
        full_audio_float32 = np.concatenate(collected_chunks)
        
        # --- NEW DIAGNOSTIC & SCALING BLOCK ---
        max_amp = np.max(np.abs(full_audio_float32))
        print(f"Diagnostic: Max audio amplitude is {max_amp:.2f}")
        
        if max_amp > 2.0:
            # The data is already scaled to the thousands! Do not multiply.
            # We just clip it to prevent resampling overshoots from causing static.
            safe_audio = np.clip(full_audio_float32, -32768.0, 32767.0)
        else:
            # --- PEAK NORMALIZATION ---
            # Divide by max_amp to bring the loudest peak to 1.0, 
            # then multiply by 0.9 to give it a 10% safety ceiling, 
            # then multiply by 32767 to convert to 16-bit integer space.
            if max_amp > 0:
                safe_audio = (full_audio_float32 / max_amp) * 0.9 * 32767.0
            else:
                safe_audio = full_audio_float32 * 32767.0
            # The data is tiny [-1.0, 1.0], so we DO need to multiply it up
            safe_audio = full_audio_float32 * 32767.0
            
        # Safely convert to 16-bit integer
        full_audio_int16 = safe_audio.astype(np.int16)
        # --------------------------------------
        
        wavfile.write(output_file, 16000, full_audio_int16)
        print(f"Success. isolated vocals here: {output_file}")
    else:
        print("No audio was collected.")