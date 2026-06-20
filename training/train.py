
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.asr_model import ASRModel
from dataset.asr_dataset import ASRDataset, collate_asr_batch
from inference.asr_decoder import ctc_decode
import csv
import os
from utils import load_metadata


os.makedirs(
    "checkpoints",
    exist_ok=True
)

#device = (

    #"mps"

    #if torch.backends.mps.is_available()

    #else "cpu"

#)

device = "cpu"

print("device:", device)



examples = load_metadata("metadata/metadata_train.csv")

print("num examples:", len(examples))
print("first example:", examples[0])
print("last example:", examples[-1])


dataset = ASRDataset(examples, use_cache = True )

loader = DataLoader(
    dataset,
    batch_size= 1,
    shuffle = True,
    num_workers = 0, 
    collate_fn=collate_asr_batch
)



model = ASRModel().to(device)


#checkpoint_path = "checkpoints/asr.pt"

#if os.path.exists(checkpoint_path):
    #model.load_state_dict(
        #torch.load(
            #checkpoint_path,
            #map_location=device
        #)
    #)

    #print("loaded existing checkpoint")


ctc = nn.CTCLoss(
    blank=0,
    zero_infinity=True
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


for epoch in range(20):

    total_loss = 0.0

    for features, targets, input_lengths, target_lengths in loader:
        
        #features = features.to(device)        
        #targets = targets.to(device)        
        
        targets = torch.cat([
            
            t[:length]
            
            for t, length in zip(
                targets,
                target_lengths
            )

        ])

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        log_probs = F.log_softmax(

            logits,
            dim=2
        )

        log_probs = log_probs.permute(
            1,
            0,
            2
        )

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

    print(epoch, avg_loss)

    if epoch % 100 == 0:

        torch.save(

            model.state_dict(),

            "checkpoints/asr.pt"

        )

        print("saved checkpoint")



print( "total time:", end - start)