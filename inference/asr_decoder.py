from dataset.asr_vocab import ID_TO_CHAR, BLANK_ID


def ctc_decode(predicted):
    
    """Converts frame-by-frame CTC predictions into readable text."""
    
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


