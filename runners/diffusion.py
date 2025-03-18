import os
import logging
import time
import glob
import cv2
import torch_dct as dct


import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from functions.denoising import generalized_steps, ddim_reverse
import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            # states = torch.load('.cache/diffusion_models_converted/ema_diffusion_cifar10_model/model-790000.ckpt')
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        model = Model(self.config)

        if self.args.use_ownmodel:
            if self.config.data.dataset == "CELEBA":
                if self.config.data.image_size == 64 :
                    states = torch.load('out/logs/bedroom-64/ckpt.pth', map_location=self.config.device)
                elif self.config.data.image_size == 128:
                    states = torch.load('out/logs/celeba-128/ckpt.pth', map_location=self.config.device)
            elif self.config.data.dataset == "CIFAR10":
                states = torch.load('out/logs/cifar-32/ckpt.pth', map_location=self.config.device)
            elif self.config.data.dataset == "LSUN":
                if self.config.data.image_size == 64 :
                    states = torch.load('out/logs/bedroom-64/ckpt.pth', map_location=self.config.device)
                elif self.config.data.image_size == 128:
                    states = torch.load('out/logs/bedroom-128/ckpt.pth', map_location=self.config.device)
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        elif self.args.reverse:
            self.sample_reverse(model)
        elif self.args.reverse_dct:
            self.sample_reverse_dct(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 1000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x_noise = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            x, _ = self.sample_image(x_noise, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        # for i in range(len(x_noise)):
        #     tvu.save_image(
        #             x_noise[i], os.path.join(self.args.image_folder, f"seq_{i}_{0}.png")
        #         )
        for i in range(len(x)):
            for j in range(1, x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"seq_{j}_{i}.png")
                )

        x_r =  torch.round(x[-1] * 255).clamp(0, 255)
        x_deq =  x_r / 127.5 - 1 
        with torch.no_grad():
            x_e, x0_pre1 = ddim_reverse(x_deq, seq, model, self.betas, eta=0.0)
        x_e = [inverse_data_transform(config, y) for y in x_e]

        for i in range(len(x_e)):
            for j in range(1, x[i].size(0)):
                tvu.save_image(
                    x_e[i][j], os.path.join(self.args.image_folder, f"seq_r{j}_{i}.png"))


    def sample_reverse(self, model):
        config = self.config
        n = 1
        mean_time = []
        Acc = []

        for ite in range(n):
            x = torch.randn(50,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                # sample image from random noise
                t1_start = time.time()
                xs = self.sample_image(x, model)
                t1_end = time.time() 
                t_gen = t1_end - t1_start

                xs_m = (xs + 1) / 2
                x0 = torch.clamp(xs_m, 0.0, 1.0)
                for i in range(len(x0)):
                    tvu.save_image(x0[i], os.path.join(self.args.image_folder, f"stego_{ite}_{i}.png"))
                
                # recover the noise from generated image x0
                x_steg =  torch.round(x0 * 255).clamp(0, 255)
                x0_dequan = x_steg / 127.5 - 1

                t2_start = time.time()
                x_t = ddim_reverse(x0_dequan, seq, model, self.betas, eta=self.args.eta)
                t_ext = time.time() - t2_start

            mean_t = (t_gen + t_ext) / (2*len(x))
            mean_time.append(mean_t)

            acc = torch.true_divide((torch.sign(x)==torch.sign(x_t[0][-1])).sum(), np.prod(x.shape))
            Acc.append(acc.item())

        diff = abs(x - x_t[0][-1])
        print('the difference of recovered latent is', diff.mean())

        print('Mean time spend is:', np.mean(mean_time), "s/img")
        print('the acc of secret data is', np.mean(Acc))
        print('Work done!')



    def sample_reverse_dct(self, model):
        config = self.config
        mean_time = []
        Acc = []
        sigma =  1.0
        ieration = 1
        num =  10

        for ite in range(ieration):
            
            secret = np.random.randint(low=0, high=2, size=(num, config.data.channels, config.data.image_size, config.data.image_size))
            z_coeff = np.zeros_like(secret, dtype=float)
            z_coeff_r = np.zeros_like(secret, dtype=float)

            coeff_m  =  (secret.astype(np.float32) * 2 - 1) * sigma

            for i in range(num):
                for j in range(config.data.channels):
                    z_coeff[i][j]= cv2.idct(coeff_m[i][j]) 


            for n in range(num):
                # 保存z_coeff的各个通道的系数保存到CSV文件中
                for i in range(config.data.channels):
                    path = os.path.join(self.args.image_folder, 'z_coeff')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    z_coeff_each = z_coeff[n][i]
                    np.savetxt(os.path.join(self.args.image_folder, f"z_coeff/z_coeff_{ite}_{n}_{i}.csv"),  z_coeff_each, delimiter="\n")


            z_s = torch.from_numpy(z_coeff).to(self.device).float()

            tvu.save_image(z_s[0:9], os.path.join(self.args.image_folder, f"zs/zs_{ite}_{0}.png"))

            with torch.no_grad():
                # 含密图像合成
                t1_start = time.time()
                xs = self.sample_image(z_s, model)
                t1_end = time.time() 
                t_gen = t1_end - t1_start

                xs_m = (xs + 1) / 2
                x0 = torch.clamp(xs_m, 0.0, 1.0)
                for i in range(len(x0)):
                    path = os.path.join(self.args.image_folder, 'stego')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    tvu.save_image(x0[i], os.path.join(self.args.image_folder, f"stego/stego_{ite}_{i}.png"))

                # 非含密图像生成
                z_ramdom = torch.randn(num,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device)
            
                x_c = self.sample_image(z_ramdom, model)
                x_c = (x_c + 1) / 2
                x_cover = torch.clamp(x_c, 0.0, 1.0)
                for i in range(len(x0)):
                    path = os.path.join(self.args.image_folder, 'cover')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    tvu.save_image(x_cover[i], os.path.join(self.args.image_folder, f"cover/cover_{ite}_{i}.png"))

                # recover the noise from generated image x0

                x_steg =  torch.round(x0 * 255).clamp(0, 255)
                x0_dequan = x_steg / 127.5 - 1

                t2_start = time.time()
                x_t = ddim_reverse(x0_dequan, seq, model, self.betas, eta=self.args.eta)
                x_r = x_t[0][-1].cpu().numpy()


                for i in range(num):
                    for j in range(config.data.channels):
                        z_coeff_r[i][j]= cv2.dct(x_r[i][j]) 

                
                t_ext = time.time() - t2_start
                mean_t = (t_gen + t_ext) / (2*len(x_r))
                mean_time.append(mean_t)

            acc = (secret.astype(np.float32) == np.ceil((np.sign(z_coeff_r) + 1) / 2) ).sum() /  np.prod(secret.shape)
            Acc.append(acc.item())

        print('Mean time spend is:', np.mean(mean_time), "s/img")
        print('the acc of secret data is', np.mean(Acc))
        print('Work done!')



    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        global seq
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                # seq = list(range(0, 10, 1)) + list(range(10, self.num_timesteps, skip))
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs

        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
