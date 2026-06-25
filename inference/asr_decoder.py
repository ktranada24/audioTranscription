from dataset.asr_vocab import ID_TO_CHAR, BLANK_ID
import torch
from collections import defaultdict
from dataset.asr_vocab import VOCAB


def ctc_decode(predicted):
    
    """"Converts frame-by-frame CTC predictions into readable text."""
    
    ids = predicted[0].tolist()
    collapsed = []

    prev = None
    
    # Collapse consecutive repeated tokens.

    for token in ids:
        if token != prev:
            collapsed.append(token)
        prev = token

    # Remove CTC blank tokens.
    collapsed = [ token for token in collapsed if token != 0]

    text = "".join(
        ID_TO_CHAR[token]
        for token in collapsed
    )

    return text


def log_sum_exp(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a

    m = max(a, b)
    return m + torch.log(torch.tensor(torch.exp(torch.tensor(a - m)) + torch.exp(torch.tensor(b - m)))).item()


def ctc_beam_decode(
    log_probs: torch.Tensor,
    beam_width: int = 10,
    blank_id: int = 0
) -> str:
    """
    Basic CTC beam search decoder.

    Args:
        log_probs: Tensor of log probabilities. Shape: (time_steps, vocab_size)
        beam_width: Number of partial hypotheses to keep.
        blank_id: CTC blank token index.

    Returns:
        str: Decoded text.
    """

    beams = {
        "": (0.0, float("-inf"))
    }
    # prefix -> (log_prob_ending_blank, log_prob_ending_nonblank)

    for t in range(log_probs.shape[0]):
        next_beams = defaultdict(
            lambda: (float("-inf"), float("-inf"))
        )

        for prefix, (p_blank, p_nonblank) in beams.items():

            for token_id in range(log_probs.shape[1]):
                p = log_probs[t, token_id].item()

                if token_id == blank_id:
                    nb_blank, nb_nonblank = next_beams[prefix]

                    nb_blank = log_sum_exp(
                        nb_blank,
                        log_sum_exp(p_blank + p, p_nonblank + p)
                    )

                    next_beams[prefix] = (
                        nb_blank,
                        nb_nonblank
                    )

                else:
                    char = VOCAB[token_id]
                    new_prefix = prefix + char

                    nb_blank, nb_nonblank = next_beams[new_prefix]

                    if len(prefix) > 0 and char == prefix[-1]:
                        # Repeated char without blank: stays same prefix.
                        same_blank, same_nonblank = next_beams[prefix]

                        same_nonblank = log_sum_exp(
                            same_nonblank,
                            p_nonblank + p
                        )

                        next_beams[prefix] = (
                            same_blank,
                            same_nonblank
                        )

                        # Repeated char after blank: creates extended prefix.
                        nb_nonblank = log_sum_exp(
                            nb_nonblank,
                            p_blank + p
                        )

                    else:
                        nb_nonblank = log_sum_exp(
                            nb_nonblank,
                            log_sum_exp(p_blank + p, p_nonblank + p)
                        )

                    next_beams[new_prefix] = (
                        nb_blank,
                        nb_nonblank
                    )

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: log_sum_exp(item[1][0], item[1][1]),
                reverse=True
            )[:beam_width]
        )

    best = max(
        beams.items(),
        key=lambda item: log_sum_exp(item[1][0], item[1][1])
    )[0]

    return best