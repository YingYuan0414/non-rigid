import os
from functools import partial
import pathlib
from typing import Dict, List, Sequence, Union, cast

import torch
import torch.utils._pytree as pytree
import torchvision as tv
import wandb
from lightning.pytorch import Callback
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf


from non_rigid.models.tax3d import (
    DiffusionTransformerNetwork,
    CrossDisplacementModule,
)
from non_rigid.models.tax3d_v2 import (
    TAX3Dv2Network,
    TAX3Dv2MuFrameModule,
    TAX3Dv2FixedFrameModule
)

from non_rigid.datasets.dedo import DedoDataModule
from non_rigid.datasets.dro import DexDataModule
from non_rigid.datasets.rigid import RigidDataModule

PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent.parent.parent.resolve())

    

def create_model(cfg):
    # setting dataset-specific model params
    cfg.model.pcd_scale = cfg.dataset.pcd_scale

    if cfg.model.name == "df_cross":
        network_fn = DiffusionTransformerNetwork
        module_fn = CrossDisplacementModule

    elif cfg.model.name == "tax3dv2":
        network_fn = TAX3Dv2Network
        if cfg.model.frame_type == "fixed":
            module_fn = TAX3Dv2FixedFrameModule
        elif cfg.model.frame_type == "mu":
            module_fn = TAX3Dv2MuFrameModule
        else:
            raise ValueError(f"Invalid frame type: {cfg.model.frame_type}")
        
    else:
        raise ValueError(f"Invalid model name: {cfg.model.name}")

    # create network and model
    network = network_fn(model_cfg=cfg.model)
    model = module_fn(network=network, cfg=cfg)

    return network, model


def create_datamodule(cfg):
    # setting model-specific dataset params
    cfg.dataset.pred_frame = cfg.model.pred_frame
    cfg.dataset.noisy_goal_scale = cfg.model.noisy_goal_scale
    cfg.dataset.action_context_frame = cfg.model.action_context_frame

    if cfg.dataset.material == "deform":
        datamodule_fn = DedoDataModule
    elif cfg.dataset.material == "rigid":
        datamodule_fn = RigidDataModule
    elif cfg.dataset.material == "hand":
        datamodule_fn = DexDataModule

    # job-specific datamodule pre-processing
    if cfg.mode == "eval":
        job_cfg = cfg.inference
        # check for full action
        if job_cfg.action_full:
            cfg.dataset.sample_size_action = -1
        stage = "predict"
    elif cfg.mode == "train":
        job_cfg = cfg.training
        stage = "fit"
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")

    # setting up datamodule
    datamodule = datamodule_fn(
        batch_size=job_cfg.batch_size,
        val_batch_size=job_cfg.val_batch_size,
        num_workers=cfg.resources.num_workers,
        dataset_cfg=cfg.dataset,
    )
    datamodule.setup(stage)

    # updating job config sample sizes
    job_cfg.sample_size = cfg.dataset.sample_size_action + cfg.dataset.sample_size_anchor
    job_cfg.sample_size_anchor = cfg.dataset.sample_size_anchor

    # training-specific job config setup
    if cfg.mode == "train":
        job_cfg.num_training_steps = len(datamodule.train_dataloader()) * job_cfg.epochs

    return cfg, datamodule

def load_checkpoint_config_from_wandb(current_cfg, task_overrides, entity, project, run_id):
    # grab run config from wandb
    api = wandb.Api()
    run_cfg = OmegaConf.create(api.run(f"{entity}/{project}/{run_id}").config)

    # check for consistency between task overrides and original run config
    inconsistent_keys = []
    for ovrd in task_overrides:
        key = ovrd.split("=")[0]
        # hack to skip data_dir overrides
        if "data_dir" in key:
            continue
        # only check for consistency with dataset/model keys
        if key.split(".")[0] not in ["dataset", "model"]:
            continue
        if OmegaConf.select(current_cfg, key) != OmegaConf.select(run_cfg, key):
            inconsistent_keys.append(key)
    
    # for now, just raise an error if there are any inconsistencies
    if inconsistent_keys:
        raise ValueError(f"Task overrides are inconsistent with original run config: {inconsistent_keys}")
    
    # hack to keep data_dir override
    current_data_dir = current_cfg.dataset.data_dir

    # update run config with dataset and model configs from original run config
    OmegaConf.update(current_cfg, "dataset", OmegaConf.select(run_cfg, "dataset"), merge=True, force_add=True)
    OmegaConf.update(current_cfg, "model", OmegaConf.select(run_cfg, "model"), merge=True, force_add=True)
    
    # small edge case - if 'eval', ignore 'train_size'/'val_size'
    if current_cfg.mode == "eval":
        current_cfg.dataset.train_size = None
        current_cfg.dataset.val_size = None
    current_cfg.dataset.data_dir = current_data_dir

    return current_cfg

# This matching function
def match_fn(dirs: Sequence[str], extensions: Sequence[str], root: str = PROJECT_ROOT):
    def _match_fn(path: pathlib.Path):
        in_dir = any([str(path).startswith(os.path.join(root, d)) for d in dirs])

        if not in_dir:
            return False

        if not any([str(path).endswith(e) for e in extensions]):
            return False

        return True

    return _match_fn


TorchTree = Dict[str, Union[torch.Tensor, "TorchTree"]]


def flatten_outputs(outputs: List[TorchTree]) -> TorchTree:
    """Flatten a list of dictionaries into a single dictionary."""

    # Concatenate all leaf nodes in the trees.
    flattened_outputs = [pytree.tree_flatten(output) for output in outputs]
    flattened_list = [o[0] for o in flattened_outputs]
    flattened_spec = flattened_outputs[0][1]  # Spec definitely should be the same...
    cat_flat = [torch.cat(x) for x in list(zip(*flattened_list))]
    output_dict = pytree.tree_unflatten(cat_flat, flattened_spec)
    return cast(TorchTree, output_dict)


class LogPredictionSamplesCallback(Callback):
    def __init__(self, logger: WandbLogger):
        self.logger = logger

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            outs = outputs["preds"][:n].argmax(dim=1)
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outs)
            ]

            # Option 1: log images with `WandbLogger.log_image`
            self.logger.log_image(key="sample_images", images=images, caption=captions)

            # Option 2: log images and predictions as a W&B Table
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred]
                for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outs))
            ]
            self.logger.log_table(key="sample_table", columns=columns, data=data)


class CustomModelPlotsCallback(Callback):
    def __init__(self, logger: WandbLogger):
        self.logger = logger

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""
        # assert trainer.logger is not None and isinstance(
        #     trainer.logger, WandbLogger
        # ), "This callback only works with WandbLogger."
        plots = pl_module.make_plots()
        trainer.logger.experiment.log(
            {
                "mode_distribution": plots["mode_distribution"],
            },
            step=trainer.global_step,
        )