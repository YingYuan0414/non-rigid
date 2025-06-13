# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import math

import numpy as np
import torch as th
import enum

from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
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
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusionMuFrame:
    """
    Utilities for training and sampling diffusion models.
    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        time_based_weighting,
        rotation_noise_scale,
    ):

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.time_based_weighting = time_based_weighting

        # TODO: custom variance for each?
        self.var_r = 1.0
        self.var_s = 1.0

        # diffusion noise scale
        self.rotation_noise_scale = rotation_noise_scale

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, xr_t, xs_t, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t) and a prediction of x₀ separately for the
        reference and shape branches.
        
        Inputs:
        - xr_t: the noisy reference input, shape [B, C, 1]
        - xs_t: the noisy shape input, shape [B, C, N]
        - t: timesteps, shape [B]
        - model: callable that takes (xr_t, xs_t, t, **model_kwargs) and returns a tuple:
                    (model_output_r, model_output_s)
        - model_kwargs: extra keyword arguments for the model.
        
        Each model_output is expected to have 2*C channels (if learning variance), where the first C
        channels predict the noise (or x₀ if ModelMeanType.START_X is used) and the next C channels are raw
        variance predictions.
        
        Returns:
        A tuple (out_r, out_s), where:
            out_r: dict with keys {"mean", "variance", "log_variance", "pred_xstart", "extra"}
                corresponding to the reference branch (shape [B, C, 1]).
            out_s: dict with the same keys for the shape branch (shape [B, C, N]).
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Get the model outputs for each branch.
        # Expected shapes:
        #   model_output_r: [B, 2C, 1]  for reference branch
        #   model_output_s: [B, 2C, N]  for shape branch
        model_output_r, model_output_s = model(xr_t, xs_t, t, **model_kwargs)
        extra = None

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # Process the reference branch.
        B, C = xr_t.shape[:2]
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output_r.shape[1] == 2 * C, f"Expected 2C channels for reference, got {model_output_r.shape[1]}"
            ref_noise, ref_var_raw = th.split(model_output_r, C, dim=1)
            # Use the spatial shape of xr_t ([B, C, 1]).
            min_log_r = _extract_into_tensor(self.posterior_log_variance_clipped, t, xr_t.shape) + math.log(self.var_r)
            max_log_r = _extract_into_tensor(np.log(self.betas), t, xr_t.shape) + math.log(self.var_r)
            frac_r = (ref_var_raw + 1) / 2  # Map from [-1, 1] to [0, 1].
            model_log_variance_r = frac_r * max_log_r + (1 - frac_r) * min_log_r
            model_variance_r = th.exp(model_log_variance_r)
        else:
            raise NotImplementedError("Non-variance learning not implemented for reference branch.")

        ##################### BUG: Below is Wrong! #####################
        # For getting pred_xstart_r, since now xr_t=denoised(GMM_pred), 
        # this is not strictly a diffusion objective, and you cannot use
        # the noise equation to revert the xstart !!!
        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart_r = process_xstart(ref_noise)
        else:
            pred_xstart_r = process_xstart(self._predict_xstart_from_eps(x_t=xr_t, t=t, eps=ref_noise))
        # Compute the branch's posterior mean using its own noisy input.
        out_r_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart_r, x_t=xr_t, t=t)
        out_r = {
            "mean": out_r_mean,
            "variance": model_variance_r,
            "log_variance": th.log(model_variance_r + 1e-8),
            "pred_xstart": pred_xstart_r,
            "extra": extra,
        }

        # Process the shape branch.
        B, C, N = xs_t.shape
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output_s.shape[1] == 2 * C, f"Expected 2C channels for shape, got {model_output_s.shape[1]}"
            shape_noise, shape_var_raw = th.split(model_output_s, C, dim=1)
            min_log_s = _extract_into_tensor(self.posterior_log_variance_clipped, t, xs_t.shape) + math.log(self.var_s)
            max_log_s = _extract_into_tensor(np.log(self.betas), t, xs_t.shape) + math.log(self.var_s)
            frac_s = (shape_var_raw + 1) / 2
            model_log_variance_s = frac_s * max_log_s + (1 - frac_s) * min_log_s
            model_variance_s = th.exp(model_log_variance_s)
        else:
            raise NotImplementedError("Non-variance learning not implemented for shape branch.")

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart_s = process_xstart(shape_noise)
        else:
            pred_xstart_s = process_xstart(self._predict_xstart_from_eps(x_t=xs_t, t=t, eps=shape_noise))
        out_s_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart_s, x_t=xs_t, t=t)
        out_s = {
            "mean": out_s_mean,
            "variance": model_variance_s,
            "log_variance": th.log(model_variance_s + 1e-8),
            "pred_xstart": pred_xstart_s,
            "extra": extra,
        }

        return out_r, out_s

    def p_mean_variance_finetune(self, model, xr_t, xs_t, t, finetune_frame, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t) and a prediction of x₀ separately for the
        reference and shape branches.
        
        Inputs:
        - xr_t: the noisy reference input, shape [B, C, 1]
        - xs_t: the noisy shape input, shape [B, C, N]
        - t: timesteps, shape [B]
        - model: callable that takes (xr_t, xs_t, t, **model_kwargs) and returns a tuple:
                    (model_output_r, model_output_s)
        - model_kwargs: extra keyword arguments for the model.
        
        Each model_output is expected to have 2*C channels (if learning variance), where the first C
        channels predict the noise (or x₀ if ModelMeanType.START_X is used) and the next C channels are raw
        variance predictions.
        
        Returns:
        A tuple (out_r, out_s), where:
            out_r: dict with keys {"mean", "variance", "log_variance", "pred_xstart", "extra"}
                corresponding to the reference branch (shape [B, C, 1]).
            out_s: dict with the same keys for the shape branch (shape [B, C, N]).
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Get the model outputs for each branch.
        # Expected shapes:
        #   model_output_r: [B, 2C, 1]  for reference branch
        #   model_output_s: [B, 2C, N]  for shape branch
        model_kwargs["finetune_frame"] = finetune_frame
        model_output_r, model_output_s = model(xr_t, xs_t, t, **model_kwargs)
        extra = None

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # Process the reference branch.
        B, C = xr_t.shape[:2]
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output_r.shape[1] == 2 * C, f"Expected 2C channels for reference, got {model_output_r.shape[1]}"
            ref_noise, ref_var_raw = th.split(model_output_r, C, dim=1)
            # Use the spatial shape of xr_t ([B, C, 1]).
            min_log_r = _extract_into_tensor(self.posterior_log_variance_clipped, t, xr_t.shape) + math.log(self.var_r)
            max_log_r = _extract_into_tensor(np.log(self.betas), t, xr_t.shape) + math.log(self.var_r)
            frac_r = (ref_var_raw + 1) / 2  # Map from [-1, 1] to [0, 1].
            model_log_variance_r = frac_r * max_log_r + (1 - frac_r) * min_log_r
            model_variance_r = th.exp(model_log_variance_r)
        else:
            raise NotImplementedError("Non-variance learning not implemented for reference branch.")

        ##################### BUG: Below is Wrong! #####################
        # For getting pred_xstart_r, since now xr_t=denoised(GMM_pred), 
        # this is not strictly a diffusion objective, and you cannot use
        # the noise equation to revert the xstart !!!
        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart_r = process_xstart(ref_noise)
        else:
            pred_xstart_r = process_xstart(self._predict_xstart_from_eps(x_t=xr_t, t=t, eps=ref_noise))
        # Compute the branch's posterior mean using its own noisy input.
        out_r_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart_r, x_t=xr_t, t=t)
        out_r = {
            "mean": out_r_mean,
            "variance": model_variance_r,
            "log_variance": th.log(model_variance_r + 1e-8),
            "pred_xstart": pred_xstart_r,
            "extra": extra,
        }

        # Process the shape branch.
        B, C, N = xs_t.shape
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output_s.shape[1] == 2 * C, f"Expected 2C channels for shape, got {model_output_s.shape[1]}"
            shape_noise, shape_var_raw = th.split(model_output_s, C, dim=1)
            min_log_s = _extract_into_tensor(self.posterior_log_variance_clipped, t, xs_t.shape) + math.log(self.var_s)
            max_log_s = _extract_into_tensor(np.log(self.betas), t, xs_t.shape) + math.log(self.var_s)
            frac_s = (shape_var_raw + 1) / 2
            model_log_variance_s = frac_s * max_log_s + (1 - frac_s) * min_log_s
            model_variance_s = th.exp(model_log_variance_s)
        else:
            raise NotImplementedError("Non-variance learning not implemented for shape branch.")

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart_s = process_xstart(shape_noise)
        else:
            pred_xstart_s = process_xstart(self._predict_xstart_from_eps(x_t=xs_t, t=t, eps=shape_noise))
        out_s_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart_s, x_t=xs_t, t=t)
        out_s = {
            "mean": out_s_mean,
            "variance": model_variance_s,
            "log_variance": th.log(model_variance_s + 1e-8),
            "pred_xstart": pred_xstart_s,
            "extra": extra,
        }

        return out_r, out_s
        
    # p_mean_variance_reverse for forward noising deltat
    def p_mean_variance_reverse(self, model, pred_ref, xr_t, xs_t, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t) and a prediction of x₀ separately for the
        reference and shape branches.
        
        Inputs:
        - xr_t: the noisy reference input, shape [B, C, 1]
        - xs_t: the noisy shape input, shape [B, C, N]
        - t: timesteps, shape [B]
        - model: callable that takes (xr_t, xs_t, t, **model_kwargs) and returns a tuple:
                    (model_output_r, model_output_s)
        - model_kwargs: extra keyword arguments for the model.
        
        Each model_output is expected to have 2*C channels (if learning variance), where the first C
        channels predict the noise (or x₀ if ModelMeanType.START_X is used) and the next C channels are raw
        variance predictions.
        
        Returns:
        A tuple (out_r, out_s), where:
            out_r: dict with keys {"mean", "variance", "log_variance", "pred_xstart", "extra"}
                corresponding to the reference branch (shape [B, C, 1]).
            out_s: dict with the same keys for the shape branch (shape [B, C, N]).
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Get the model outputs for each branch.
        # Expected shapes:
        #   model_output_r: [B, 2C, 1]  for reference branch
        #   model_output_s: [B, 2C, N]  for shape branch
        model_output_r, model_output_s = model(pred_ref, xs_t, t, **model_kwargs)
        extra = None

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # Process the reference branch.
        B, C = xr_t.shape[:2]
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output_r.shape[1] == 2 * C, f"Expected 2C channels for reference, got {model_output_r.shape[1]}"
            ref_noise, ref_var_raw = th.split(model_output_r, C, dim=1)
            # Use the spatial shape of xr_t ([B, C, 1]).
            min_log_r = _extract_into_tensor(self.posterior_log_variance_clipped, t, xr_t.shape) + math.log(self.var_r)
            max_log_r = _extract_into_tensor(np.log(self.betas), t, xr_t.shape) + math.log(self.var_r)
            frac_r = (ref_var_raw + 1) / 2  # Map from [-1, 1] to [0, 1].
            model_log_variance_r = frac_r * max_log_r + (1 - frac_r) * min_log_r
            model_variance_r = th.exp(model_log_variance_r)
        else:
            raise NotImplementedError("Non-variance learning not implemented for reference branch.")

        if self.model_mean_type == ModelMeanType.START_X:
            raise NotImplementedError
            #pred_xstart_r = process_xstart(ref_noise)
        else:
            #pred_xstart_r = process_xstart(self._predict_xstart_from_eps(x_t=xr_t, t=t, eps=ref_noise))
            # NOTE: we inpaint delta_0 = 0 here, and we re-calculate delta_t using q_sample
            pred_deltastart = th.zeros_like(xr_t)
            pred_delta_t = self.q_sample(x_start=pred_deltastart, t=t, noise=ref_noise)
            
            
        # Compute the branch's posterior mean using its own noisy input.
        # NOTE: so here, the output is for the delta, not z!
        out_r_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_deltastart, x_t=pred_delta_t, t=t)

        if (t == self.num_timesteps - 1).any():
            prev_delta_t = pred_delta_t
        else:
            prev_delta_t = xr_t

        out_r = {
            "mean": out_r_mean,
            "variance": model_variance_r,
            "log_variance": th.log(model_variance_r + 1e-8),
            "pred_xstart": pred_deltastart,
            "extra": extra,
            "prev_delta_t": prev_delta_t
        }

        # Process the shape branch.
        B, C, N = xs_t.shape
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output_s.shape[1] == 2 * C, f"Expected 2C channels for shape, got {model_output_s.shape[1]}"
            shape_noise, shape_var_raw = th.split(model_output_s, C, dim=1)
            min_log_s = _extract_into_tensor(self.posterior_log_variance_clipped, t, xs_t.shape) + math.log(self.var_s)
            max_log_s = _extract_into_tensor(np.log(self.betas), t, xs_t.shape) + math.log(self.var_s)
            frac_s = (shape_var_raw + 1) / 2
            model_log_variance_s = frac_s * max_log_s + (1 - frac_s) * min_log_s
            model_variance_s = th.exp(model_log_variance_s)
        else:
            raise NotImplementedError("Non-variance learning not implemented for shape branch.")

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart_s = process_xstart(shape_noise)
        else:
            pred_xstart_s = process_xstart(self._predict_xstart_from_eps(x_t=xs_t, t=t, eps=shape_noise))
        out_s_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart_s, x_t=xs_t, t=t)
        out_s = {
            "mean": out_s_mean,
            "variance": model_variance_s,
            "log_variance": th.log(model_variance_s + 1e-8),
            "pred_xstart": pred_xstart_s,
            "extra": extra,
        }

        return out_r, out_s

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape, f"Error: x_t.shape {x_t.shape} != eps.shape {eps.shape}"
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model,
        pred_ref,
        x_r,
        x_s,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        
        Inputs:
        - model: the model to sample from.
        - x_r: the current noisy reference input, shape [B, C, 1]
        - x_s: the current noisy shape input, shape [B, C, N]
        - t: timesteps, shape [B]
        - clip_denoised: if True, clip the x_start prediction to [-1, 1].
        - denoised_fn: optional function to post-process the x_start prediction.
        - cond_fn: an optional conditioning function (e.g. for guidance).
        - model_kwargs: additional keyword arguments for the model.
        
        Returns:
        A dict containing:
            - 'sample': a sample from p(x_{t-1} | x_t).
            - 'pred_xstart': the combined prediction for x₀.
        """
        import torch as th

        # Obtain separate outputs for the reference and shape branches.
        out_r, out_s = self.p_mean_variance_reverse(
            model,
            pred_ref,
            x_r,
            x_s,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        noise_r = th.randn_like(x_r)
        noise_s = th.randn_like(x_s)
        nonzero_mask_r = (
            (t != 0).float().view(-1, *([1] * (len(x_r.shape) - 1)))
        )  # no noise when t == 0
        nonzero_mask_s = (
            (t != 0).float().view(-1, *([1] * (len(x_s.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out_r["mean"] = self.condition_mean(cond_fn, out_r, x_r, t, model_kwargs=model_kwargs)
            out_s["mean"] = self.condition_mean(cond_fn, out_s, x_s, t, model_kwargs=model_kwargs)

        sample_r = out_r["mean"] + nonzero_mask_r * th.exp(0.5 * out_r["log_variance"]) * noise_r
        sample_s = out_s["mean"] + nonzero_mask_s * th.exp(0.5 * out_s["log_variance"]) * noise_s

        return {"sample_r": sample_r, "sample_s": sample_s, "prev_sample_r": out_r["prev_delta_t"]}

    def p_sample_finetune(
        self,
        model,
        finetune_frame,
        x_r,
        x_s,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        
        Inputs:
        - model: the model to sample from.
        - x_r: the current noisy reference input, shape [B, C, 1]
        - x_s: the current noisy shape input, shape [B, C, N]
        - t: timesteps, shape [B]
        - clip_denoised: if True, clip the x_start prediction to [-1, 1].
        - denoised_fn: optional function to post-process the x_start prediction.
        - cond_fn: an optional conditioning function (e.g. for guidance).
        - model_kwargs: additional keyword arguments for the model.
        
        Returns:
        A dict containing:
            - 'sample': a sample from p(x_{t-1} | x_t).
            - 'pred_xstart': the combined prediction for x₀.
        """
        import torch as th

        # Obtain separate outputs for the reference and shape branches.
        out_r, out_s = self.p_mean_variance_finetune(
            model,
            x_r,
            x_s,
            t,
            finetune_frame,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        noise_r = th.randn_like(x_r)
        noise_s = th.randn_like(x_s)
        nonzero_mask_r = (
            (t != 0).float().view(-1, *([1] * (len(x_r.shape) - 1)))
        )  # no noise when t == 0
        nonzero_mask_s = (
            (t != 0).float().view(-1, *([1] * (len(x_s.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out_r["mean"] = self.condition_mean(cond_fn, out_r, x_r, t, model_kwargs=model_kwargs)
            out_s["mean"] = self.condition_mean(cond_fn, out_s, x_s, t, model_kwargs=model_kwargs)

        sample_r = out_r["mean"] + nonzero_mask_r * th.exp(0.5 * out_r["log_variance"]) * noise_r
        sample_s = out_s["mean"] + nonzero_mask_s * th.exp(0.5 * out_s["log_variance"]) * noise_s

        return {"sample_r": sample_r, "sample_s": sample_s, "pred_xstart": out_r["pred_xstart"]+out_s["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape_r,
        shape_s,
        noise_r=None,
        noise_s=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        results = [{"sample_r": noise_r, "sample_s": noise_s}]  # NOTE: for t=T, samplr_r is actually in global frame
        pred_ref = noise_r.clone()
        for out in self.p_sample_loop_progressive(
            model,
            shape_r,
            shape_s,
            noise_r=noise_r,
            noise_s=noise_s,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = out
            # NOTE: from t=T-1, sample_r will be the action_mean_err
            results.append({"sample_r": final["sample_r"], "sample_s": final["sample_s"], "prev_sample_r":final["prev_sample_r"]})
        final_dict = {"sample_r": final["sample_r"], "sample_s": final["sample_s"], "prev_sample_r":final["prev_sample_r"]}
        return final_dict, results

    def p_sample_loop_progressive(
        self,
        model,
        shape_r,
        shape_s,
        noise_r=None,
        noise_s=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape_r, (tuple, list))
        assert isinstance(shape_s, (tuple, list))
        if noise_r is not None:
            img_r = noise_r
        else:
            img_r = th.randn(*shape_r, device=device)
        if noise_s is not None:
            img_s = noise_s
        else:
            img_s = th.randn(*shape_s, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        
        pred_ref = noise_r.clone()

        # 1. delta_t = 0
        #img_r = th.zeros_like(pred_ref)
        # 2. delta_t ~ N(0,1)
        # TODO
        # 3. delta_t ~ N(0,1) but is included within GMM_pred
        # Initial delta_t from q(δ_T | δ₀=0)
        '''
        t_init = th.tensor([self.num_timesteps - 1] * pred_ref.shape[0], device=pred_ref.device)
        eps = th.randn_like(pred_ref)
        delta_t = self.q_sample(x_start=th.zeros_like(pred_ref), t=t_init, noise=eps)
        img_r = delta_t.clone()
        img_s = noise_s if noise_s is not None else th.randn(*shape_s, device=device)
        '''
        img_r = th.zeros_like(pred_ref)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape_s[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    pred_ref,
                    img_r,
                    img_s,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                delta_t_prev = out["prev_sample_r"]  # This is delta_{t-1}
                delta_t = out["sample_r"]
                pred_ref = pred_ref - delta_t_prev + delta_t   # z_{t-1} = z_t - δ_t + δ_{t-1}

                img_s = out["sample_s"]
                img_r = out["sample_r"]
                out["sample_r"] = pred_ref

                yield out

    def p_sample_loop_finetune(
        self,
        model,
        shape_r,
        shape_s,
        noise_r=None,
        noise_s=None,
        finetune_frame=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        assert finetune_frame is not None, "Please set finetune frame!"
        final = None
        results = [{"sample_r": noise_r, "sample_s": noise_s}]  # NOTE: for t=T, samplr_r is actually in global frame
        pred_ref = noise_r.clone()
        for out in self.p_sample_loop_progressive_finetune(
            model,
            shape_r,
            shape_s,
            noise_r=noise_r,
            noise_s=noise_s,
            finetune_frame=finetune_frame,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = out
            # NOTE: from t=T-1, sample_r will be the action_mean_err
            results.append({"sample_r": final["sample_r"], "sample_s": final["sample_s"]})
        final_dict = {"sample_r": final["sample_r"], "sample_s": final["sample_s"]}
        return final_dict, results

    def p_sample_loop_progressive_finetune(
        self,
        model,
        shape_r,
        shape_s,
        noise_r=None,
        noise_s=None,
        finetune_frame=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        assert finetune_frame is not None, "Please set finetune frame!"
        assert noise_r is not None, "Please set img_r!"
        assert noise_s is not None, "Please set img_s!"

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape_r, (tuple, list))
        assert isinstance(shape_s, (tuple, list))

        img_r = noise_r
        img_s = noise_s

        finetune_steps = range(10)
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            finetune_steps = tqdm(finetune_steps)

        for i in finetune_steps:
            t = th.tensor([0] * shape_s[0], device=device)
            with th.no_grad():
                out = self.p_sample_finetune(
                    model,
                    finetune_frame,
                    img_r,
                    img_s,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img_r = out["sample_r"]
                img_s = out["sample_s"]


    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(self, model, xr_start, xs_start, xr_t, xs_t, t, clip_denoised=True, model_kwargs=None):
        """
        Compute variational bound terms for the combined diffusion process by first computing the
        true posterior for the reference and shape branches individually and then fusing the results.

        Inputs:
        - xr_start: ground-truth reference component, shape [B, C, 1]
        - xs_start: ground-truth shape component, shape [B, C, N]
        - xr_t: noisy reference input, shape [B, C, 1]
        - xs_t: noisy shape input, shape [B, C, N]
        - t: timesteps, shape [B]
        - clip_denoised: whether to clip the denoised prediction.
        - model_kwargs: additional kwargs for the model.

        Returns:
        A dict with keys:
            "output": a [B]-shaped tensor of KL divergences (or decoder NLL at t = 0, in bits),
            "pred_xstart": the combined model prediction for x₀.
        """
        out_r, out_s = self.p_mean_variance(model, xr_t, xs_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)

        # vb for r (reference)
        true_mean_r, _, true_log_variance_clipped_r = self.q_posterior_mean_variance(
            x_start=xr_start, x_t=xr_t, t=t)
        

        kl_r = normal_kl(
            true_mean_r, true_log_variance_clipped_r, out_r["mean"], out_r["log_variance"]
        )
        kl_r = mean_flat(kl_r) / np.log(2.0)

        decoder_nll_r = -discretized_gaussian_log_likelihood(
            xr_start, means=out_r["mean"], log_scales=0.5 * out_r["log_variance"]
        )
        assert decoder_nll_r.shape == xr_start.shape
        decoder_nll_r = mean_flat(decoder_nll_r) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output_r = th.where((t == 0), decoder_nll_r, kl_r)

        # vb for s (shape)
        true_mean_s, _, true_log_variance_clipped_s = self.q_posterior_mean_variance(
            x_start=xs_start, x_t=xs_t, t=t)
        
        kl_s = normal_kl(
            true_mean_s, true_log_variance_clipped_s, out_s["mean"], out_s["log_variance"]
        )
        kl_s = mean_flat(kl_s) / np.log(2.0)

        decoder_nll_s = -discretized_gaussian_log_likelihood(
            xs_start, means=out_s["mean"], log_scales=0.5 * out_s["log_variance"]
        )
        assert decoder_nll_s.shape == xs_start.shape
        decoder_nll_s = mean_flat(decoder_nll_s) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output_s = th.where((t == 0), decoder_nll_s, kl_s)

        # combine outputs
        #pred_xstart_combined = out_r["pred_xstart"].expand(-1, -1, xs_t.shape[-1]) + out_s["pred_xstart"]

        #return {"output_r": output_r, "output_s": output_s, "pred_xstart": pred_xstart_combined}
        return {"output_r": output_r, "output_s": output_s}

    def _time_based_weights(self, t, T, func="sigmoid", k=10.0, m=0.5):
        """
        Compute time-based weights using a sigmoid function.

        Parameters:
            t (float or Tensor): Current timestep (0 <= t <= T)
            T (float): Total number of timesteps
            func (string): Type of time-based function
            k (float): Slope parameter of the sigmoid (default: 10.0)
            m (float): Midpoint of the sigmoid (default: 0.5)

        Returns:
            w_r, w_s (tuple): Where w_s = sigmoid coefficient, and w_r = 1 - w_s.
        """
        # Ensure t is a float tensor
        if not th.is_tensor(t):
            t = th.tensor(t, dtype=th.float64)

        x = t / T  # Normalize to [0, 1]

        if func == "even":
            coeff = th.tensor(0.5, dtype=th.float64, device=x.device)
        elif func == "linear":
            coeff = x
        elif func == "sigmoid":
            # Make sure everything is on the same device and is a tensor
            x = x.float()
            k = th.tensor(k, dtype=th.float64, device=x.device)
            m = th.tensor(m, dtype=th.float64, device=x.device)

            Lx = 1.0 / (1.0 + th.exp(-k * (x - m)))
            L0 = 1.0 / (1.0 + th.exp(-k * (0.0 - m)))
            L1 = 1.0 / (1.0 + th.exp(-k * (1.0 - m)))
            coeff = (Lx - L0) / (L1 - L0)
        else:
            raise ValueError(f"Unsupported function type {func}. Use 'sigmoid', 'linear', or 'even'.")

        w_s = coeff
        w_r = 1.0 - coeff
        return w_r, w_s

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep using separate forward processes for
        the reference (R) and shape (S) components.
        
        x_start: [B, C, N] ground-truth point cloud.
        Assumes that the reference component is given by the global translation (e.g. mean over points)
        and the shape component is defined as the residual.
        
        In this design, we sample independent noise for each branch, run their q_sample forward processes,
        and then compute losses (either KL/variational bound or MSE) on each branch separately.
        The overall prediction x̂₀ is obtained as R̂₀ + Ŝ₀.
        """
        import torch as th
        if model_kwargs is None:
            model_kwargs = {}

        # Decompose the ground-truth into reference and shape parts.
        # Here, we take the reference as the mean (global translation), and the shape as the residual.
        xr_start = x_start.mean(dim=-1, keepdim=True)  # [B, C, 1]
        xs_start = x_start - xr_start                    # [B, C, N]
        xr_start_err = th.zeros_like(xr_start)

        # Sample independent noise terms for each branch.
        noise_r_err = th.randn_like(xr_start_err)  # noise for R: [B, C, 1]
        noise_s = th.randn_like(xs_start)  # noise for S: [B, C, N]

        if self.rotation_noise_scale:
            N, C, P = x_start.shape  # e.g. [batch_size, 3, num_points]

            # 1. Sample random rotation axes and normalize them.
            random_axis = th.randn(N, 3, device=x_start.device)
            random_axis = random_axis / random_axis.norm(dim=1, keepdim=True)

            # 2. Sample random rotation angles (in degrees) and convert to radians.
            random_angle = th.randn(N, device=x_start.device) * self.rotation_noise_scale
            random_angle_rad = random_angle * (np.pi / 180)

            # 3. Compute sin and cos for each angle.
            sin_theta = th.sin(random_angle_rad)
            cos_theta = th.cos(random_angle_rad)
            one_minus_cos = 1 - cos_theta

            # 4. Extract axis components.
            x = random_axis[:, 0]
            y = random_axis[:, 1]
            z = random_axis[:, 2]
            zeros = th.zeros_like(x)

            # 5. Construct the skew-symmetric cross-product matrices for each axis.
            #    Each K is of shape (3,3), and K will have shape (N, 3, 3)
            K = th.stack([th.stack([zeros, -z, y], dim=1),
                        th.stack([z, zeros, -x], dim=1),
                        th.stack([-y, x, zeros], dim=1)], dim=1)

            # 6. Compute K squared (batched matrix multiplication)
            K2 = th.bmm(K, K)

            # 7. Create the identity matrix for each batch element.
            I = th.eye(3, device=x_start.device).unsqueeze(0).repeat(N, 1, 1)

            # 8. Compute the rotation matrices using the Rodrigues formula:
            #    R = I + sin(theta)*K + (1-cos(theta))*(K^2)
            R = I + sin_theta.view(N, 1, 1) * K + one_minus_cos.view(N, 1, 1) * K2

            # 9. Apply the rotation matrices to the point clouds.
            #    x_start: [N, 3, P] -> rotated_pc: [N, 3, P]
            rotated_pc = th.bmm(R, x_start)

            # 10. The rotation noise is the difference between the rotated and original points.
            rotation_noise = rotated_pc - x_start
            noise_s = noise_s + rotation_noise

        # Compute the forward noising for each branch.
        xr_err_t = self.q_sample(xr_start_err, t, noise=noise_r_err)  # [B, C, 1]
        xs_t = self.q_sample(xs_start, t, noise=noise_s)  # [B, C, N]

        # Center our anchor on the noised goal
        pred_ref = xr_err_t + xr_start
        #model_kwargs["y"] = model_kwargs["y"] - pred_ref

        terms = {}
        if self.loss_type in [LossType.KL, LossType.RESCALED_KL]:
            raise NotImplementedError

        elif self.loss_type in [LossType.MSE, LossType.RESCALED_MSE]:
            # For MSE training, assume that the model is designed to process each branch separately.
            # Here we feed the branch-specific noisy inputs to the model.
            model_output_r, model_output_s = model(pred_ref, xs_t, t, **model_kwargs)

            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                # Process branch R: split into mean prediction and variance.
                B, C = xr_err_t.shape[:2]
                assert model_output_r.shape == (B, C * 2, 1), f"model_output_r: {model_output_r.shape}"
                model_output_r, model_var_values_r = th.split(model_output_r, C, dim=1)
                frozen_out_r = th.cat([model_output_r.detach(), model_var_values_r], dim=1)

                # Process branch S.
                B, C, N = xs_t.shape
                assert model_output_s.shape == (B, C * 2, N), f"model_output_s: {model_output_s.shape}"
                model_output_s, model_var_values_s = th.split(model_output_s, C, dim=1)
                frozen_out_s = th.cat([model_output_s.detach(), model_var_values_s], dim=1)


                frozen_out = (frozen_out_r, frozen_out_s)

                vb_bpd_dict = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    xr_start=xr_start_err,
                    xs_start=xs_start,
                    xr_t=xr_err_t,
                    xs_t=xs_t,
                    t=t,
                    clip_denoised=False,
                )
                terms["vb_r"] = vb_bpd_dict["output_r"]
                terms["vb_s"] = vb_bpd_dict["output_s"]
                terms["vb"] = vb_bpd_dict["output_r"] + vb_bpd_dict["output_s"]

                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0


            # Ensure that the shapes of the model outputs match the noise targets.
            assert model_output_r.shape == noise_r_err.shape, f"model_output_r {model_output_r.shape} not equal to noise_r {noise_r_err.shape}"
            assert model_output_s.shape == noise_s.shape, f"model_output_s {model_output_s.shape} not equal to noise_s {noise_s.shape}"

            loss_r = ((noise_r_err - model_output_r) ** 2).mean()
            loss_s = ((noise_s - model_output_s) ** 2).mean()

            # Optionally, apply time-based weighting.
            w_r, w_s = self._time_based_weights(t=t, T=self.num_timesteps, func=self.time_based_weighting)
            terms["mse_r"] = w_r * loss_r
            terms["mse_s"] = w_s * loss_s
            terms["mse"] = w_r * loss_r + w_s * loss_s

            if "vb" in terms:
                terms["loss_r"] = terms["mse_r"] + terms["vb_r"]
                terms["loss_s"] = terms["mse_s"] + terms["vb_s"]
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss_r"] = terms["mse_r"]
                terms["loss_s"] = terms["mse_s"]
                terms["loss"] = terms["mse"]

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)
