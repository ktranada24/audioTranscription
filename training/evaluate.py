from inference.decode import load_model, transcribe_audio
from training.utils import load_metadata
import random
from dataset.asr_vocab import VOCAB

model = load_model()

examples = load_metadata("metadata/metadata_val.csv")



def edit_distance(a: str, b: str) -> int:

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):

        dp[i][0] = i

    for j in range(len(b) + 1):

        dp[0][j] = j

    for i in range(1, len(a) + 1):

        for j in range(1, len(b) + 1):

            if a[i - 1] == b[j - 1]:

                cost = 0

            else:

                cost = 1

            dp[i][j] = min(

                dp[i - 1][j] + 1,        # deletion

                dp[i][j - 1] + 1,        # insertion

                dp[i - 1][j - 1] + cost  # substitution

            )

    return dp[len(a)][len(b)]


def character_error_rate(truth: str, pred: str) -> float:

    if len(truth) == 0:

        return 0.0 if len(pred) == 0 else 1.0

    return edit_distance(truth, pred) / len(truth)


total_cer = 0

for audio, truth in examples:
    
    pred = transcribe_audio(audio, model)
   
    cer = character_error_rate(truth, pred)
    
    total_cer += cer

    print("truth:", truth)

    print("pred :", pred)

    print("CER  :", cer)

    print()    
    
avg_cer = total_cer / len(examples)



VOCAB_CHARS = [
    token
    for token in VOCAB
    if token != "<blank>"
]

def random_prediction(length: int) -> str:

    return "".join(

        random.choice(VOCAB_CHARS)

        for _ in range(length)

    )

trials = 100

trial_cers = []

for _ in range(trials):

    total_cer = 0.0

    for audio_path, truth in examples:

        pred = random_prediction(len(truth))

        total_cer += character_error_rate(truth, pred)

    trial_cers.append(total_cer / len(examples))

baseline_cer = sum(trial_cers) / len(trial_cers)

print("random baseline CER mean:", baseline_cer)

print("average CER:", avg_cer)

print("Skill score:", 1 - (avg_cer/baseline_cer))

    
   