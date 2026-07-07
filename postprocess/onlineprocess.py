
common_mispellings = {
    "dont": "don't",
    "doesnt": "doesn't",
    "didnt": "didn't",
    "cant": "can't",
    "wont": "won't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "im": "I'm",
    "ive": "I've",
    "ill": "I'll",
    "id": "I'd",
    "youre": "you're",
    "theyre": "they're",
    "weve": "we've",
    "thats": "that's",
    "whats": "what's",
}


def postprocess_online(text: str):
    text = text.strip()
    text = text.replace('i', 'I')
    text = text.split()
    for i in range(len(text)):
        if text[i] in common_mispellings:
            text[i] = common_mispellings[text[i]]
    text = ' '.join(text)
    
    return text

print(postprocess_online('cant do   it today sorry mann i am sorry   '))

    
    
    
