from inference.decode import load_model, transcribe_features, transcribe_features_beam
import random
from dataset.asr_vocab import VOCAB
from dataset.asr_dataset import LibriSpeechASRDataset

model = load_model()

eval_dataset = LibriSpeechASRDataset(
    start=500,
    limit=100,
    use_cache=True,
    return_transcript=True

)

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

for i in range(len(eval_dataset)):

    features, target_ids, truth = eval_dataset[i]

    pred = transcribe_features(

    features,

    model)

    cer = character_error_rate(truth, pred)

    total_cer += cer
    
    print("truth:", truth)    
    print("pred :", pred)    
    print("CER  :", cer)
    print()    


avg_cer =  total_cer / len(eval_dataset)
print("average CER:", avg_cer)


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

trials = 10

trial_cers = []

for _ in range(trials):

    total_cer = 0.0

    for i in range(len(eval_dataset)):
        
        features, target_ids, truth = eval_dataset[i]

        pred = random_prediction(len(truth))

        total_cer += character_error_rate(truth, pred)

    trial_cers.append(total_cer / len(eval_dataset))

baseline_cer = sum(trial_cers) / len(trial_cers)

print("random baseline CER mean:", baseline_cer)

print("average CER:", avg_cer)

print("Skill score:", 1 - (avg_cer/baseline_cer))
