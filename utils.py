import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from scipy.spatial import distance

def train(
    model,
    config,
    train_loader,
    valid_loader=None,# you can add validation process to test the OOD situation
    valid_epoch_interval=5,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    # best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        print("training process")
        print('current epoch:',epoch_no)
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )
def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))
def calc_quantile_CRPS(target, forecast, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, 1)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], 1)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(model, myscaler, test_loader, nsample=1, scaler=1, mean_scaler=0, foldername=""):
    print('testnsample=:',nsample)
    with torch.no_grad():
        model.eval()
        evalpoints_total = 0
        all_target = []
        all_observed_time = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                sample, c_targets, observed_time = output
                sample = sample.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                a, b, c, d = sample.shape
                c_targets = c_targets.permute(0, 2, 1)  # (B,L,K)
                x, y, z = c_targets.shape
                sample = torch.tensor(
                    myscaler.inverse_transform(sample.reshape(-1, 1).detach().cpu().numpy()).reshape(a, b, c, d))
                c_targets = torch.tensor(
                    myscaler.inverse_transform(c_targets.reshape(-1, 1).detach().cpu().numpy()).reshape(x, y, z))

                samples = sample
                c_target = c_targets
                B, L, K = c_target.shape

                all_target.append(c_target)

                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)
                evalpoints_total += (B*K*L)


            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, mean_scaler, scaler
            )


            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        CRPS,
                    ],
                    f,
                )
                print("CRPS:", CRPS)
