from decode import load_model, transcribe_audio
from utils import load_metadata

model = load_model()

examples = load_metadata("metadata_train.csv")

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


for audio, truth in examples:
    
    pred = transcribe_audio(audio, model)
    
    print("truth:", truth)
    print("pred :", pred)    