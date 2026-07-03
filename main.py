from inference.decode import load_model
from pipeline.online import run_online
from pipeline.offline import run_offline

MODE = "online"

model = load_model("checkpoints/best_val.pt")

if MODE == "online":
    run_online(model)

elif MODE == "offline":
    run_offline("audio.wav", model)