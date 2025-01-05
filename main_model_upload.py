
import torch
import torch.nn as nn
from diff_model import STKDiff


class STK_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.mse = nn.MSELoss()
        config_diff = config["diffusion"]
        input_dim = 1
        self.diffmodel = STKDiff(config_diff, input_dim)
        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = torch.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = torch.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        self.alpha = 1. - self.beta.to('cuda')
        self.alpha_hat = torch.cumprod(self.alpha,dim=0).to('cuda')


    def calc_loss_valid(
        self, observed_data, is_train, idx
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, is_train, idx, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps
    def noise_images(self, x, t):
        t = t.to(self.alpha_hat.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x).to(self.alpha_hat.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    def calc_loss(
        self, observed_data,  is_train, idx, set_t=-1
    ):
        B, K, L = observed_data.shape

        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        noisy_data, noise = self.noise_images(observed_data,t)

        total_input = self.set_input_to_diffmodel(noisy_data)#here

        predicted = self.diffmodel(total_input, t, idx)  # (B,K,L)
        loss =self.mse(noise,predicted)
        return loss

    def set_input_to_diffmodel(self, noisy_data):
        total_input = noisy_data  # (B,K,L,1)
        return total_input

    def impute(self, observed_data, idx,n_samples):

        B, K, L = observed_data.shape

        imputed_samples = torch.zeros( B, n_samples,K, L).to(self.device)

        for i in range(n_samples):
            x = torch.randn_like(observed_data)

            for t in reversed(range(1, self.num_steps)):
                diff_input = x
                predicted = self.diffmodel(diff_input, torch.tensor([t]).to(self.device), idx)
                alpha = self.alpha[t].unsqueeze(-1).unsqueeze(-1).to(self.device)
                alpha_hat = self.alpha_hat[t].unsqueeze(-1).unsqueeze(-1).to(self.device)
                beta = self.beta[t].unsqueeze(-1).unsqueeze(-1).to(self.device)

                if t > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted) + torch.sqrt(
                    beta) * noise
            record_sample = x


            imputed_samples[:,i] = record_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_tp,
            idx,
        ) = self.process_data(batch)


        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, is_train, idx)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_tp,
            idx,
        ) = self.process_data(batch)

        with torch.no_grad():
            samples = self.impute(observed_data,idx, n_samples)
        return samples, observed_data,observed_tp


class STK_model(STK_base):
    def __init__(self, config, device, target_dim=1):
        super(STK_model, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        idx = batch["idx"].to(self.device).int()
        observed_data = observed_data.permute(0, 2, 1)


        return (
            observed_data,
            observed_tp,
            idx,
        )
