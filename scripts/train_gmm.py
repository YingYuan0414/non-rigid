import numpy as np
import torch
import omegaconf
import json
import os
import shutil
import hydra
import wandb

from plotly import graph_objects as go

from tqdm import tqdm
import wandb.util

from non_rigid.utils.script_utils import create_datamodule, create_model
from non_rigid.models.gmm_predictor import FrameGMMPredictor, GMMLoss, viz_gmm



# ignore TypedStorage warnings
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)

@hydra.main(config_path="../configs", config_name="train_gmm", version_base="1.3")
def main(cfg):
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    # Global seed for reproducibility.
    torch.manual_seed(cfg.seed)

    # Setting device.
    device = f"cuda:{cfg.resources.gpus[0]}"

    ######################################################################
    # Create the datamodule.
    ######################################################################
    cfg, datamodule = create_datamodule(cfg)
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    ######################################################################
    # Create the network.
    ######################################################################
    network, _ = create_model(cfg)
    model = FrameGMMPredictor(network, cfg.model, device)

    ######################################################################
    # Training loop.
    ######################################################################
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = GMMLoss(cfg)
    num_epochs = cfg.epochs
    val_every = num_epochs // 10

    # Training statistics.
    total_losses = []
    total_val_losses = []
    total_probs99 = []
    total_probs90 = []
    total_probs50 = []

    # Creating experiment directory.
    run_id = wandb.util.generate_id(length=10) # For simplicity, using wandb.util.generate_id() as a unique identifier.
    exp_name = os.path.join(
        os.path.expanduser(cfg.gmm_log_dir),
        run_id,
    )

    if os.path.exists(exp_name):
        print(f"Experiment directory {exp_name} already exists. Removing it.")
        shutil.rmtree(exp_name)
    os.makedirs(exp_name, exist_ok=True)
    os.makedirs(os.path.join(exp_name, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_name, "logs"), exist_ok=True)

    # Saving config.
    with open(os.path.join(exp_name, "config.yaml"), "w") as f:
        omegaconf.OmegaConf.save(cfg, f)

    # Visualizing initial model.
    fig, num_probs99, num_probs90, num_probs50 = viz_gmm(model, train_dataset)
    fig.update_layout(title_text="Epoch 0")
    fig.write_html(os.path.join(exp_name, "logs", f"epoch_0.html"))
    total_probs99.append(num_probs99)
    total_probs90.append(num_probs90)
    total_probs50.append(num_probs50)

    # Training loop.
    print(f"Training GMM with run ID: {run_id}")
    with tqdm(total=num_epochs) as pbar:
        pbar.set_description("Training")
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = []

            # Training step.
            for i, batch in enumerate(train_loader):
                # Compute model prediction.
                optimizer.zero_grad()
                pred = model(batch)

                # Compute and backprop loss.
                loss = loss_fn(batch, pred, var=cfg.var, uniform_loss=cfg.uniform_loss, regularize_residual=cfg.regularize_residual)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Update statistics.
                epoch_loss.append(loss.item())
            total_losses.append(np.mean(epoch_loss))

            # Validation step.
            if (epoch + 1) % val_every == 0:
                model.eval()
                val_loss = []
                with torch.no_grad():
                    for i, batch in enumerate(val_loader):
                        # Compute model prediction.
                        pred = model(batch)

                        # Compute loss.
                        loss = loss_fn(batch, pred, var=cfg.var, uniform_loss=cfg.uniform_loss, regularize_residual=cfg.regularize_residual)
                        val_loss.append(loss.item())
                total_val_losses.append(np.mean(val_loss))

                # Save model checkpoint.
                torch.save(model.state_dict(), os.path.join(exp_name, "checkpoints", f"epoch_{epoch + 1}.pt"))

                # Also log visualizations.
                val_fig, num_probs99, num_probs90, num_probs50 = viz_gmm(model, train_dataset)
                val_fig.update_layout(
                    title_text=f"Epoch {epoch + 1}, Train Loss: {total_losses[-1]:.4f}, Val Loss: {total_val_losses[-1]:.4f}",
                )
                val_fig.write_html(os.path.join(exp_name, "logs", f"epoch_{epoch + 1}.html"))
                total_probs99.append(num_probs99)
                total_probs90.append(num_probs90)
                total_probs50.append(num_probs50)

                # Update progress bar.
                pbar.set_postfix(
                    train_loss=total_losses[-1],
                    val_loss=total_val_losses[-1],
                )
            else:
                pbar.set_postfix(
                    train_loss=total_losses[-1],
                )
            pbar.update(1)

    # After training, plot the losses.
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, len(total_losses) + 1), y=total_losses, name="Train Loss"))
    fig.add_trace(go.Scatter(x=np.arange(val_every, (len(total_val_losses) + 1) * val_every, val_every), y=total_val_losses, name="Val Loss"))
    fig.update_layout(title="Losses", xaxis_title="Epoch", yaxis_title="Loss")
    fig.write_html(os.path.join(exp_name, "logs", "losses.html"))

    # AFter training, plot probability statistics.
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(val_every, (len(total_val_losses) + 1) * val_every, val_every), y=total_probs99, name="Top-0.99"))
    fig.add_trace(go.Scatter(x=np.arange(val_every, (len(total_val_losses) + 1) * val_every, val_every), y=total_probs90, name="Top-0.90"))
    fig.add_trace(go.Scatter(x=np.arange(val_every, (len(total_val_losses) + 1) * val_every, val_every), y=total_probs50, name="Top-0.50"))
    fig.update_layout(title="Top-K Probabilities", xaxis_title="Epoch", yaxis_title="Number of Points")
    fig.write_html(os.path.join(exp_name, "logs", "top_probs.html"))

if __name__ == "__main__":
    main()