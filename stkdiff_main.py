
import argparse
import torch
import datetime
import json
import yaml
import os
from main_model_upload import STK_model
from dataset_loader import get_dataloader
from utils import train,  evaluate

os.environ['CUDA_VISIBLE_DEVICES']='5'


parser = argparse.ArgumentParser(description="STKDiff")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)


print(torch.__version__)

args = parser.parse_args()



cityname = 'bj'
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/physio_fold"  + "_" + current_time + cityname + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, myscaler = get_dataloader(
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
)

model = STK_model(config, args.device).to(args.device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
if args.modelfolder == "":
    print('begin train')
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, myscaler, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
