import csv
import torchaudio
import torch
import torchaudio


model_config = {
    "input_size": 80,
    "hidden_size": 256,
    "num_layers": 2,
    "vocab_size": 29,
}


def load_metadata(csv_path: str):

    examples = []

    with open(csv_path, "r") as f:

        reader = csv.DictReader(f)

        for row in reader:
            examples.append(
                (row["audio_path"],
                row["transcript"]
                ))
            
    return examples


def save_checkpoint(path, epoch, model, optimizer, avg_loss, best_val_cer=None):
    torch.save(
        {
            "epoch": epoch,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "best_val_cer": best_val_cer,
        },
        path
    )
    
    
def get_librispeech_split_range(  
    root: str,  
    url: str,  
    start_percent: float,  
    end_percent: float) -> tuple[int, int]:
    
    full_dataset = torchaudio.datasets.LIBRISPEECH( 
        root=root,  
        url=url,  
        download=False  
    )
    
    total = len(full_dataset) 
    start = int(total * start_percent)    
    end = int(total * end_percent)
    
    limit = end - start
    
    return start, limit