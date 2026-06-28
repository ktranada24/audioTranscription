import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.asr_model import ASRModel
from dataset.asr_dataset import ASRDataset, LibriSpeechASRDataset, collate_asr_batch
from inference.asr_decoder import ctc_decode
import csv
import os
from training.utils import save_checkpoint ,  get_librispeech_split_range
import time
from training.evaluate import eval_diagnostics,  eval_dataset_val


os.makedirs("checkpoints", exist_ok=True)

# Compute Device Allocation
mode = "hybrid"  # "cpu" or "hybrid"

# Train-set range
start_percent = 0.0
end_percent = 0.95

# Save-settings
use_cache = True
load_progress = False
load_best = False

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


start, limit =  get_librispeech_split_range(root="data/librispeech",
    url="train-clean-100",
    start_percent= start_percent,
    end_percent= end_percent)


model = ASRModel().to(device)


dataset = LibriSpeechASRDataset(
    root="data/librispeech",
    url="train-clean-100",
    start= start,
    limit= limit,
    use_cache=True,
)


loader = DataLoader(
    dataset,
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers = num_workers, 
    collate_fn=collate_asr_batch)


print("dataset size:", len(dataset))


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = learn_rate,
    weight_decay= weight_decay
)

if load_progress:
    if load_best and os.path.exists('checkpoints/best_val.pt'):
        model.load_state_dict(torch.load('checkpoints/best_val.pt', map_location=device))
        checkpoint = torch.load("checkpoints/best_val.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        best_val_cer = checkpoint["best_val_cer"]
        start_epoch = checkpoint["epoch"] + 1           
        print("loaded best existing checkpoint")
        
    elif os.path.exists('checkpoints/latest.pt'):
        checkpoint = torch.load("checkpoints/latest.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        best_val_cer = checkpoint["best_val_cer"]
        start_epoch = checkpoint["epoch"] + 1        
        print("loaded latest existing checkpoint")


ctc = nn.CTCLoss(blank=0, zero_infinity=True)


start_time = time.time()

best_val_cer = None
patience_counter = 0

for epoch in range(num_epoch):

    total_loss = 0.0

    for features, targets, input_lengths, target_lengths in loader:
        
        features = features.to(device)
          
        targets = torch.cat([   
            t[:length] for t, length in zip( targets, target_lengths)])

        optimizer.zero_grad(set_to_none=True)
        
        logits = model(features)
        log_probs = F.log_softmax(logits, dim=2)

        log_probs = log_probs.permute(1, 0, 2)

        if mode == "hybrid":
            loss = ctc(
                log_probs.cpu(),
                targets.cpu(),
                input_lengths.cpu(),
                target_lengths.cpu() )
    
        else:
            loss = ctc(
                log_probs,
                targets,
                input_lengths,
                target_lengths)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm = clip_grad_maxnorm)
        
        optimizer.step()        
        
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f'Itiration {epoch}:',f'loss {avg_loss}') 

    if epoch % 1 == 0:    
        save_checkpoint(
            "checkpoints/latest.pt",
            epoch,
            model,
            optimizer,
            avg_loss,
            best_val_cer
        )
        print("saved checkpoint")
        
        if epoch % 3 == 0:
            
            current_cer = eval_diagnostics(eval_dataset_val, inspect_predictions = False, skill_score = False)
        
            if best_val_cer == None:
                best_val_cer = current_cer
                save_checkpoint(
                    "checkpoints/best_val.pt",
                    epoch,
                    model,
                    optimizer,
                    avg_loss,
                    best_val_cer)
                print("saved best checkpoint")
                
                
            elif current_cer < best_val_cer:
                best_val_cer = current_cer
                save_checkpoint(
                    "checkpoints/best_val.pt",
                    epoch,
                    model,
                    optimizer,
                    avg_loss,
                    best_val_cer)
                print("saved new best checkpoint")
                patience_counter = 0
                
            else:
                patience_counter += 1
                print(f'patience count: {patience_counter}')
                if patience_counter >= 5:
                    break 
            
            
end_time = time.time()     
print("total training time:", end_time - start_time)


