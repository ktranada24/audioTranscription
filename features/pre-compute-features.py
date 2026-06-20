import torch

import os

from dataset.asr_dataset import (

    load_audio_file,
    waveform_to_log_mel

)

examples = [

(

    "test.wav",

    "hello my name is nikhil"

)

]

os.makedirs(

    "cache",

    exist_ok=True

)

for audio_path, transcript in examples:

    print("processing:", audio_path)

    waveform = load_audio_file(
        audio_path

    )

    waveform = waveform.numpy()
    
    features = waveform_to_log_mel(waveform)

    save_path = ("cache/" + audio_path + ".pt")

    torch.save(features, save_path)

print("done")