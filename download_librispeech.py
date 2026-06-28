import torchaudio

dataset = torchaudio.datasets.LIBRISPEECH(
    root="data/librispeech",
    url="train-clean-100",
    download=True
)

print("num samples:", len(dataset))

waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = dataset[0]

print(waveform.shape)
print(sample_rate)
print(transcript)