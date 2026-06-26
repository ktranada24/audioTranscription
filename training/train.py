import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.asr_model import ASRModel
from dataset.asr_dataset import ASRDataset, LibriSpeechASRDataset, collate_asr_batch
from inference.asr_decoder import ctc_decode
import csv
import os
from training.utils import load_metadata
import time 


os.makedirs("checkpoints", exist_ok=True)


# Compute Device Allocation
mode = "hybrid"  # "cpu" or "hybrid"

# Train-set range
start = 0
limit = 500

# Save-settings
use_cache = True
load_progress = False

# Optimization-settings
batch_size = 4
num_workers = 0
shuffle = True
learn_rate = 0.001
weight_decay = 1e-4
clip_grad_maxnorm = 5.0

# Training length 
num_epoch = 81

# etc
return_transcript = False


if mode == "hybrid" and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("mode:", mode)
print("device:", device)


model = ASRModel().to(device)


dataset = LibriSpeechASRDataset(
    root="data/librispeech",
    url="dev-clean",
    start =start, 
    limit=limit,
    use_cache = use_cache,  
    return_transcript = return_transcript)


loader = DataLoader(
    dataset,
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers = num_workers, 
    collate_fn=collate_asr_batch)


print("dataset size:", len(dataset))


if load_progress:
    checkpoint_path = "checkpoints/asr.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
    print("loaded existing checkpoint")


ctc = nn.CTCLoss(blank=0, zero_infinity=True)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = learn_rate,
    weight_decay= weight_decay
)


start_time = time.time()


for epoch in range(num_epoch):

    total_loss = 0.0

    for features, targets, input_lengths, target_lengths in loader:
        
        features = features.to(device)
          
        targets = torch.cat([   
            t[:length]   
            for t, length in zip( targets, target_lengths)
        ])

        optimizer.zero_grad(set_to_none=True)
        
        logits = model(features)
        log_probs = F.log_softmax(logits, dim=2)

        log_probs = log_probs.permute(1, 0, 2)

        if mode == "hybrid":
            loss = ctc(
                log_probs.cpu(),
                targets.cpu(),
                input_lengths.cpu(),
                target_lengths.cpu()
            )
    
        else:
            loss = ctc(
                log_probs,
                targets,
                input_lengths,
                target_lengths
            )

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )
        
        optimizer.step()        
        
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f'Itiration {epoch}:',f'loss {avg_loss}')

    if epoch % 10 == 0:    
        torch.save( model.state_dict(), "checkpoints/asr.pt")
        print("saved checkpoint")


end_time = time.time()     
print("total training time:", end_time - start_time)


