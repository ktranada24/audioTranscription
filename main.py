from inference.decode import load_model
from pipeline.online import run_online
from pipeline.offline import run_offline

MODE = "online"
model = load_model("checkpoints/best_val.pt")

def main():
        
    if MODE == "online":
        model = load_model("checkpoints/best_val.pt")
        run_online("audio.wav", model)
    
    elif MODE == "offline":
        model = load_model("checkpoints/best_val.pt")
        run_offline("audio.wav", model)
        
        
if __name__ == "__main__":
    main()