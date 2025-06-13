import numpy as np
import torch
from omegaconf import OmegaConf
import json
import os
import shutil
import hydra

from plotly import graph_objects as go

from tqdm import tqdm

from non_rigid.utils.script_utils import create_datamodule, create_model
from non_rigid.models.gmm_predictor import FrameGMMPredictor

import rpad.visualize_3d.plots as rvpl
import matplotlib.pyplot as plt

# ignore TypedStorage warnings
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)

@hydra.main(config_path="../configs", config_name="eval_gmm", version_base="1.3")
def main(cfg):
    # Grab config file from saved run.
    exp_name = os.path.join(
        os.path.expanduser(cfg.gmm_log_dir),
        cfg.checkpoint.run_id,
    )
    if not os.path.exists(exp_name):
        raise ValueError(f"Experiment directory {exp_name} does not exist - train this model first.")
    run_cfg = OmegaConf.load(os.path.join(exp_name, "config.yaml"))

    # Update config with run config.
    OmegaConf.update(cfg, "dataset", OmegaConf.select(run_cfg, "dataset"), merge=True, force_add=True)
    OmegaConf.update(cfg, "model", OmegaConf.select(run_cfg, "model"), merge=True, force_add=True)
    print(
        json.dumps(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            sort_keys=True,
            indent=4,
        )
    )

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
    # Manually setting eval-specific configs.
    ######################################################################
    # Using a custom cloth-specific batch size, to allow for simultaneous evaluation 
    # of RMSE, coverage, and precision.
    if cfg.dataset.hole == "single":
        bs = 1
    elif cfg.dataset.hole == "double":
        bs = 2
    else:
        raise ValueError(f"Unknown hole type: {cfg.dataset.hole}.")
    bs *= cfg.dataset.num_anchors

    ######################################################################
    # Create the datamodule.
    ######################################################################
    if cfg.dataset.name != "dedo":
        raise ValueError(f"This evaluation script only works with the DEDO dataset.")
    cfg, datamodule = create_datamodule(cfg)
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset

    ######################################################################
    # Create the network.
    ######################################################################
    network, _ = create_model(cfg)
    model = FrameGMMPredictor(network, cfg.model, device)
    model.eval()

    ######################################################################
    # Evaluation loop.
    ######################################################################
    # For now, only evaluate the last checkpoint.
    viz_path = os.path.join(exp_name, "viz")
    ckpt_path = os.path.join(exp_name, "checkpoints", f"epoch_{run_cfg.epochs}.pt")

    os.makedirs(os.path.join(viz_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(viz_path, "val"), exist_ok=True)

    # Loading model checkpoint.
    model.load_state_dict(torch.load(ckpt_path))

    # Helper function to compute evaluation metrics.
    @torch.no_grad()
    def eval_dataset(dataset, model, split):
        rmses = []
        coverage_rmses = []
        precision_rmses = []
        num_batches = len(dataset) // bs
        data_keys = ["pc_action", "pc_anchor", "pc"]

        for i in tqdm(range(num_batches), desc=f"Evaluating {split} set", leave=False):
            gt_means = []
            sampled_means = []

            for j in range(bs):
                item_j = dataset[i * bs + j]
                item_j = {key: value.unsqueeze(0) for key, value in item_j.items() if key in data_keys}
                pred_j = model(item_j)

                # Computing ground truth action mean in anchor frame.
                anchor_frame = pred_j["anchor_frame"].cpu()
                targets = item_j["pc"] - anchor_frame
                targets = targets.mean(dim=1)[0].cpu()
                gt_means.append(targets.unsqueeze(0))

                # Sampling predictions.
                probs, means = pred_j["probs"], pred_j["means"]
                idxs = torch.multinomial(probs.squeeze(-1), cfg.num_samples // bs, replacement=True).squeeze()
                sampled_means.append(means[:, idxs].squeeze())

            # Concatenating all samples.
            gt_means = torch.cat(gt_means, dim=0).to(device)
            sampled_means = torch.cat(sampled_means, dim=0)

            # Compute pairwise RMSEs, and update statistics.
            dists = torch.cdist(gt_means, sampled_means, p=2)
            rmses.append(dists.flatten().cpu().numpy())
            coverage_rmses.append(dists.min(dim=1).values.cpu().numpy())
            precision_rmses.append(dists.min(dim=0).values.cpu().numpy())
            
            # Plot point clouds.
            fig = go.Figure()
            anchor_pc = (item_j["pc_anchor"] - anchor_frame)[0].cpu().numpy()
            probs_viz = probs.squeeze(-1).cpu().numpy().reshape((-1,))
            means_viz = means.cpu().numpy().reshape((-1, 3))
            gt_means_pc = gt_means.cpu().numpy()
            sampled_means_pc = sampled_means.cpu().numpy()

            scene_data = np.concatenate(
                [anchor_pc, gt_means_pc, sampled_means_pc, means_viz], axis=0
            )

            fig.add_trace(
                go.Scatter3d(
                    mode="markers",
                    marker=dict(size=6, color=probs_viz, colorscale="Inferno", cmin=0),
                    x=anchor_pc[:, 0],
                    y=anchor_pc[:, 1],
                    z=anchor_pc[:, 2],
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    mode="markers",
                    marker=dict(size=6, color="red"),
                    x=gt_means_pc[:, 0],
                    y=gt_means_pc[:, 1],
                    z=gt_means_pc[:, 2],
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    mode="markers",
                    marker=dict(size=6, color="green"),
                    x=sampled_means_pc[:, 0],
                    y=sampled_means_pc[:, 1],
                    z=sampled_means_pc[:, 2],
                )
            )

            # Visualizing predicted means.
            fig.add_trace(
                go.Scatter3d(
                    mode="markers",
                    marker=dict(size=6, color=probs_viz, colorscale="Inferno", cmin=0,
                                symbol="circle-open", opacity=0.5),
                    x=means_viz[:, 0],
                    y=means_viz[:, 1],
                    z=means_viz[:, 2],
                    name="Pred Means",
                )
            )
            
            # Adding vectors between anchor point cloud and predicted means.
            # x_lines, y_lines, z_lines = [], [], []
            # for anchor_point, mean_point in zip(anchor_pc, means_viz):
            #     x_lines += [anchor_point[0], mean_point[0], None]
            #     y_lines += [anchor_point[1], mean_point[1], None]
            #     z_lines += [anchor_point[2], mean_point[2], None]

            # fig.add_trace(go.Scatter3d(
            #         x=[None],
            #         y=[None],
            #         z=[None],
            #         mode='lines',
            #         line=dict(color='black', width=4),
            #         name='A → B vectors',
            #         showlegend=True
            #     ))
            norm = plt.Normalize(vmin=0, vmax=probs_viz.max())
            cmap = plt.cm.get_cmap("inferno")
            colors = [f'rgba{cmap(norm(p), bytes=True)}' for p in probs_viz]

            vector_trace_indices = []
            for k, (anchor_point, mean_point) in enumerate(zip(anchor_pc, means_viz)):
                fig.add_trace(go.Scatter3d(
                    x=[anchor_point[0], mean_point[0]],
                    y=[anchor_point[1], mean_point[1]],
                    z=[anchor_point[2], mean_point[2]],
                    mode='lines',
                    line=dict(color=colors[k], width=4),
                    showlegend=True,
                    legendgroup="vectors",
                    visible=True,
                ))
                vector_trace_indices.append(k)
            # fig.add_trace(
            #     go.Scatter3d(
            #         mode="lines",
            #         line=dict(color=probs_viz, colorscale="Inferno", cmin=0, width=4),
            #         x=x_lines,
            #         y=y_lines,
            #         z=z_lines,
            #         name="Anchor to Pred Means",
            #     )
            # )

            fig.update_layout(
                scene=rvpl._3d_scene(scene_data),
            )
            fig.update_scenes(
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False,
            )

            fig.write_html(os.path.join(viz_path, split, f"{i}.html"))

        # Plot statistics.
        rmses = np.concatenate(rmses)
        coverage_rmses = np.concatenate(coverage_rmses)
        precision_rmses = np.concatenate(precision_rmses)
        hist = go.Figure()
        hist.add_trace(
            go.Histogram(
                x=rmses,
                name="RMSE",
            )
        )
        hist.add_trace(
            go.Histogram(
                x=precision_rmses,
                name="Precision RMSE",
            )
        )
        hist.add_trace(
            go.Histogram(
                x=coverage_rmses,
                name="Coverage RMSE",
            )
        )
        hist.update_layout(barmode="overlay")
        hist.update_traces(opacity=0.75)
        hist.update_layout(title_text=f"{split} Histograms")
        hist.write_html(os.path.join(viz_path, split, "hist.html"))

        rmses = np.mean(rmses)
        coverage_rmses = np.mean(coverage_rmses)
        precision_rmses = np.mean(precision_rmses)
        print(f"RMSE: {rmses}, Coverage RMSE: {coverage_rmses}, Precision RMSE: {precision_rmses}")


    # Eavluating datasets.
    eval_dataset(train_dataset, model, "train")
    eval_dataset(val_dataset, model, "val")

if __name__ == "__main__":
    main()