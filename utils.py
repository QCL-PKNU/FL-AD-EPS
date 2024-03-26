import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.dlutils import ComputeLoss
from src.datafolders import data_folder, output_folder
from src.constants import *
from src.utils import *


def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size : i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w.view(-1))
    return torch.stack(windows)


def load_dataset(dataset):
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception("Processed Data not found.")
    loader = []
    for file in ["train", "test", "labels"]:
        loader.append(np.load(os.path.join(folder, f"{file}.npy")))
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f"checkpoints/{args.name}/"
    os.makedirs(folder, exist_ok=True)
    file_path = f"{folder}/model.ckpt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "accuracy_list": accuracy_list,
        },
        file_path,
    )


def load_model(modelname, dims):
    import src.models

    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

    if args.use_fl:
        fname = f"checkpoints/{args.model}_{args.dataset}_FL/model.ckpt"
    else:
        fname = f"checkpoints/{args.model}_{args.dataset}/model.ckpt"

    if args.client is not None:
        fname = f"checkpoints/{args.model}_{args.dataset}{args.client}/model.ckpt"

    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        accuracy_list = checkpoint["accuracy_list"]
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def load_dataset_partition(idx):
    folder = os.path.join(output_folder, "EPS", f"partition_{idx}")
    if not os.path.exists(folder):
        raise Exception("Processed Data not found.")
    loader = []
    for file in ["train", "test", "labels"]:
        loader.append(np.load(os.path.join(folder, f"{file}.npy")))
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction="mean" if training else "none")
    feats = dataO.shape[1]
    if "DAGMM" in model.name:
        l = nn.MSELoss(reduction="none")
        compute = ComputeLoss(model, 0.1, 0.005, "cpu", model.n_gmm)
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        l2s = []
        if training:
            for d in data:
                _, x_hat, z, gamma = model(d)
                l1, l2 = l(x_hat, d), l(gamma, d)
                l1s.append(torch.mean(l1).item())
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1) + torch.mean(l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f"Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}")
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]["lr"]
        else:
            ae1s = []
            for d in data:
                _, x_hat, _, _ = model(d)
                ae1s.append(x_hat)
            ae1s = torch.stack(ae1s)
            y_pred = ae1s[:, data.shape[1] - feats : data.shape[1]].view(-1, feats)
            loss = l(ae1s, data)[:, data.shape[1] - feats : data.shape[1]].view(
                -1, feats
            )
            return loss.detach().numpy(), y_pred.detach().numpy()
    if "Attention" in model.name:
        l = nn.MSELoss(reduction="none")
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        res = []
        if training:
            for d in data:
                ae, ats = model(d)
                # res.append(torch.mean(ats, axis=0).view(-1))
                l1 = l(ae, d)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
            scheduler.step()
            tqdm.write(f"Epoch {epoch},\tL1 = {np.mean(l1s)}")
            return np.mean(l1s), optimizer.param_groups[0]["lr"]
        else:
            ae1s, y_pred = [], []
            for d in data:
                ae1 = model(d)
                y_pred.append(ae1[-1])
                ae1s.append(ae1)
            ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
            loss = torch.mean(l(ae1s, data), axis=1)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif "OmniAnomaly" in model.name:
        if training:
            mses, klds = [], []
            for i, d in enumerate(data):
                y_pred, mu, logvar, hidden = model(d, hidden if i else None)
                MSE = l(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + model.beta * KLD
                mses.append(torch.mean(MSE).item())
                klds.append(model.beta * torch.mean(KLD).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f"Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}")
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]["lr"]
        else:
            y_preds = []
            for i, d in enumerate(data):
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.detach().numpy(), y_pred.detach().numpy()
    elif "USAD" in model.name:
        l = nn.MSELoss(reduction="none")
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d in data:
                ae1s, ae2s, ae2ae1s = model(d)
                l1 = (1 / n) * l(ae1s, d) + (1 - 1 / n) * l(ae2ae1s, d)
                l2 = (1 / n) * l(ae2s, d) - (1 - 1 / n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item())
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f"Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}")
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]["lr"]
        else:
            ae1s, ae2s, ae2ae1s = [], [], []
            for d in data:
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1)
                ae2s.append(ae2)
                ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = (
                torch.stack(ae1s),
                torch.stack(ae2s),
                torch.stack(ae2ae1s),
            )
            y_pred = ae1s[:, data.shape[1] - feats : data.shape[1]].view(-1, feats)
            loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1] - feats : data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif "AE" in model.name:
        l = nn.MSELoss(reduction="none")
        l1 = nn.L1Loss(reduction="none")
        n = epoch + 1
        l1s = []
        if training:
            for _, d in enumerate(data):
                x = model(d)
                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f"Epoch {epoch + 1},\tMSE = {np.mean(l1s)}")
            return np.mean(l1s), optimizer.param_groups[0]["lr"]
        else:
            xs = []
            for d in data:
                x = model(d)
                xs.append(x)
            xs = torch.stack(xs)
            y_pred = xs[:, data.shape[1] - feats : data.shape[1]].view(-1, feats)
            loss = l(xs, data)
            loss = loss[:, data.shape[1] - feats : data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif model.name in ["GDN", "MTAD_GAT", "MSCRED", "CAE_M"]:
        l = nn.MSELoss(reduction="none")
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        if training:
            for i, d in enumerate(data):
                if "MTAD_GAT" in model.name:
                    x, h = model(d, h if i else None)
                else:
                    x = model(d)
                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f"Epoch {epoch},\tMSE = {np.mean(l1s)}")
            return np.mean(l1s), optimizer.param_groups[0]["lr"]
        else:
            xs = []
            for d in data:
                if "MTAD_GAT" in model.name:
                    x, h = model(d, None)
                else:
                    x = model(d)
                xs.append(x)
            xs = torch.stack(xs)
            y_pred = xs[:, data.shape[1] - feats : data.shape[1]].view(-1, feats)
            loss = l(xs, data)
            loss = loss[:, data.shape[1] - feats : data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif "GAN" in model.name:
        l = nn.MSELoss(reduction="none")
        bcel = nn.BCELoss(reduction="mean")
        msel = nn.MSELoss(reduction="mean")
        real_label, fake_label = torch.tensor([0.9]), torch.tensor(
            [0.1]
        )  # label smoothing
        real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(
            torch.DoubleTensor
        )
        n = epoch + 1
        w_size = model.n_window
        mses, gls, dls = [], [], []
        if training:
            for d in data:
                # training discriminator
                model.discriminator.zero_grad()
                _, real, fake = model(d)
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()
                model.generator.zero_grad()
                optimizer.step()
                # training generator
                z, _, fake = model(d)
                mse = msel(z, d)
                gl = bcel(fake, real_label)
                tl = gl + mse
                tl.backward()
                model.discriminator.zero_grad()
                optimizer.step()
                mses.append(mse.item())
                gls.append(gl.item())
                dls.append(dl.item())
                # tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
            tqdm.write(
                f"Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}"
            )
            return np.mean(gls) + np.mean(dls), optimizer.param_groups[0]["lr"]
        else:
            outputs = []
            for d in data:
                z, _, _ = model(d)
                outputs.append(z)
            outputs = torch.stack(outputs)
            y_pred = outputs[:, data.shape[1] - feats : data.shape[1]].view(-1, feats)
            loss = l(outputs, data)
            loss = loss[:, data.shape[1] - feats : data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    else:
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f"Epoch {epoch},\tMSE = {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]["lr"]
        else:
            return loss.detach().numpy(), y_pred.detach().numpy()
