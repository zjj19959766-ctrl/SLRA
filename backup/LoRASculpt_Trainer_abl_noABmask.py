# ==========================ZJJ: 代码更新优化主要在这个py文件中==========================
# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================
# [消融实验] 无 AB_masks 幅度稀疏化
# 基于：LoRASculpt_Trainer3.1.py
#
# 修改内容（仅此一处）：
#   - 删除 self.AB_masks = {} 初始化
#   - 删除 STEP_THRESHOLD 时刻创建掩码的 if 块（幅度 Top-K 稀疏化）
#   - 删除 restore_lora_param 函数定义及每步调用循环
#
# 保留内容（与 v3.1 完全一致）：
#   - comput_custom_reg 的 HOG+幅度融合正则化 (CMR_LAMBDA)
#   - ResidualPatch 补丁训练（动态 epoch、Cosine 调度、梯度累积、HOG 约束）
#   - SVD 吸收（合并 Patch 到主 LoRA 后保存）
#
# 实验目的：
#   隔离 AB_masks 幅度稀疏化机制的独立贡献，
#   与完整 v3.1（AB_masks + CMR + Patch + SVD）对比，
#   说明稀疏化在 LoRA 参数空间中的去噪作用。
# ============================================================


import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from deepspeed.utils import \
    safe_get_full_fp32_param, safe_get_local_fp32_param, \
    safe_get_full_grad, safe_get_local_grad,\
    safe_set_full_fp32_param,safe_set_local_fp32_param

# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from llava.train.llava_trainer import LLaVATrainer

import torch.nn.functional as F


from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    get_dataloader_sampler,
    get_model_param_count,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.utils.quantization_config import QuantizationMethod

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


def _is_peft_model(model):
    return is_peft_available() and isinstance(model, PeftModel)


if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"



STEP_THRESHOLD=int(os.environ.get('STEP_THRESHOLD', 100))
AB_PRESERVE_RATIO=float(os.environ.get('AB_PRESERVE_RATIO', 0.1))
# 获取环境变量，默认值设为 0.5 (你 1.py 里的数值)
HOG_LAMBDA = float(os.environ.get('HOG_LAMBDA', 0.5))
CMR_LAMBDA=float(os.environ.get('CMR_LAMBDA', 1e-3))
OMEGA=float(os.environ.get('OMEGA', 1.0))



# 1. 补丁类：极小的 LoRA，专门用于修补残差

# 1. 补丁类：极小的 LoRA，专门用于修补残差 (终极无痛版)
class ResidualPatch(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=0.25, hog_lambda=0.5, omega=1.0, base_weight=None):
        super().__init__()
        self.patch_A = nn.Parameter(torch.zeros(rank, in_features))
        self.patch_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.hog_lambda = hog_lambda
        self.omega = omega

        nn.init.kaiming_uniform_(self.patch_A, a=math.sqrt(5))
        nn.init.zeros_(self.patch_B)

        # === [终极优化] 拒绝囤积矩阵！只存对原权重的轻量级"引用"(占用0显存) ===
        self.base_weight = base_weight

    def forward(self, x, original_output):
        dtype = x.dtype
        device = x.device
        p_A = self.patch_A.to(device)
        p_B = self.patch_B.to(device)
        x = x.to(p_A.dtype)
        patch_out = (x @ p_A.T @ p_B.T) * self.scaling
        return original_output + patch_out.to(dtype)

    # === [新增] 计算 Patch 自身的正则化损失 ===
    def get_reg_loss(self):
        # === [终极优化] 即用即算，算完马上释放！===
        if self.base_weight is not None:
            with torch.no_grad():
                hog_prior = compute_hog_prior(self.base_weight)
                hog_min = hog_prior.min()
                hog_max = hog_prior.max()
                hog_norm = (hog_prior - hog_min) / (hog_max - hog_min + 1e-8)
                del hog_prior # 阅后即焚

                w_norm = torch.norm(self.base_weight.to(torch.float32), p=2).to(self.base_weight.dtype)
                M_pre = self.base_weight / (w_norm + 1e-8)
                S_mag = torch.abs((1 / torch.log(M_pre.abs() + 1e-15)))
                del M_pre # 阅后即焚

                S_combined = S_mag + self.hog_lambda * hog_norm
                del S_mag, hog_norm # 阅后即焚

                hog_mask = torch.tanh(self.omega * S_combined)
                del S_combined # 阅后即焚

            # 算完 Mask 马上乘，乘完马上把 Mask 扔掉
            delta_W = (self.patch_B @ self.patch_A) * self.scaling
            loss = torch.norm(hog_mask * delta_W, p=2)
            del hog_mask

            return loss
        return 0.0
    # [优化] 用于推理阶段的无痕合并
    def merge_to_base(self, base_layer):
        if self.patch_A is not None and self.patch_B is not None:
            with torch.no_grad():
                delta_W = (self.patch_B @ self.patch_A) * self.scaling
                base_layer.weight.data += delta_W.to(base_layer.weight.dtype)



# 2. HOG 计算函数：放在外面，供 comput_custom_reg 调用
_SOBEL_CACHE = {} # 全局缓存
def compute_hog_prior(weight_tensor):
    if len(weight_tensor.shape) == 2:
        w_img = weight_tensor.unsqueeze(0).unsqueeze(0)
    else:
        return torch.abs(weight_tensor)

    device = weight_tensor.device
    dtype = weight_tensor.dtype

    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3)

    grad_x = F.conv2d(w_img, sobel_x, padding=1)
    grad_y = F.conv2d(w_img, sobel_y, padding=1)

    # 【注意看这里！是逗号不是加号！】
    magnitude = torch.hypot(grad_x, grad_y) + 1e-8
    return magnitude.squeeze()

class LoRASculpt(LLaVATrainer):

    # ... [放在 LoRASculpt 类内部] ...

    def mount_residual_patches(self, model, patch_rank=4):
        """
        [综合优化] 定向挂载：直接狙击核心多模态对齐层
        """
        logger.info("🚑 Mounting Patches on targeted bottleneck layers...")
        target_layer_keywords = ["mm_projector", "v_proj", "q_proj"]
        patches, trainable_params = [], []

        # 冻结所有基础参数
        for p in model.parameters():
            p.requires_grad = False

        for name, module in model.named_modules():
            if any(k in name for k in target_layer_keywords) and isinstance(module, nn.Linear):
                patch_alpha = float(os.environ.get('PATCH_ALPHA', 1.0))
                patch = ResidualPatch(
                    module.in_features, module.out_features,
                    rank=patch_rank, alpha=patch_alpha,
                    hog_lambda=HOG_LAMBDA, omega=OMEGA, base_weight=module.weight
                ).to(module.weight.device)

                def hook_fn(mod, inp, out, p=patch):
                    return p(inp[0], out)

                module.register_forward_hook(hook_fn)
                patches.append(patch)
                trainable_params.extend(list(patch.parameters()))
                logger.info(f"   ➕ Target Patch mounted on: {name}")

        self.active_patches = patches
        return trainable_params




    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):

        # [消融] 移除 self.AB_masks = {} 初始化，不再追踪任何稀疏掩码


        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {'use_reentrant':False}


            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1



            for step, inputs in enumerate(epoch_iterator):

                # [消融] 移除 STEP_THRESHOLD 时刻的 Top-K 幅度掩码创建块
                # [消融] 移除 restore_lora_param 函数定义及每步调用循环
                # 所有 LoRA 参数在整个训练过程中自由更新，不施加稀疏约束

                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model): # automatically perform the gradient accumulation
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):  # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):  # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping
                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    self.optimizer.step()



                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)



                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)
        # ============================================================
        # [新增] 第三步：自动触发 LoRA of LoRA 补丁训练
        # ============================================================
        # ============================================================
        # [改进一] 第三步：按 Epoch 比例自动触发 LoRA of LoRA 补丁训练
        # ============================================================
        # ============================================================
        # ============================================================
        # [新增] 第三步：自动触发 LoRA of LoRA 补丁训练
        # ============================================================
        try:
            p_rank = int(os.environ.get('PATCH_RANK', 8))
            patch_params = self.mount_residual_patches(self.model, patch_rank=p_rank)

            if patch_params:
                grad_acc_steps = self.args.gradient_accumulation_steps

                # 动态读取 epoch，计算步数
                patch_epochs = float(os.environ.get('PATCH_TRAIN_EPOCHS', 0.5))
                # 确保这里拿到的是真实的 dataloader 长度
                steps_per_epoch = len(train_dataloader) if len_dataloader is not None else 2500
                refine_steps = int((steps_per_epoch * patch_epochs) / grad_acc_steps)
                refine_steps = max(refine_steps, 1)

                logger.info(f"🚀 Starting Targeted Refinement for {patch_epochs} epochs ({refine_steps} steps)...")

                # 【关键修复】：绝对不把主 LoRA 放进原生优化器，避免 DeepSpeed 冲突！
                # 只让 Patch 自己独立、快速地学习残差
                patch_optimizer = torch.optim.AdamW([
                    {'params': patch_params, 'lr': 5e-4}
                ])

                import transformers
                scheduler = transformers.get_cosine_schedule_with_warmup(
                    patch_optimizer, num_warmup_steps=int(refine_steps * 0.1), num_training_steps=refine_steps
                )

                self.model.train()
                refine_iterator = iter(train_dataloader)

                for step in range(refine_steps):
                    try:
                        inputs = next(refine_iterator)
                    except StopIteration:
                        refine_iterator = iter(train_dataloader)
                        inputs = next(refine_iterator)

                    inputs = self._prepare_inputs(inputs)

                    # 避免使用 compute_loss_context_manager，防止深层反向传播冲突
                    outputs = self.model(**inputs)
                    loss = outputs["loss"]

                    # 加入 Patch 的独立正则化约束
                    if hasattr(self, 'active_patches'):
                        patch_reg_loss = sum(p.get_reg_loss() for p in self.active_patches)
                        loss += CMR_LAMBDA * patch_reg_loss

                    loss = loss / grad_acc_steps
                    loss.backward()

                    if (step + 1) % grad_acc_steps == 0 or (step + 1) == refine_steps:
                        patch_optimizer.step()
                        scheduler.step()
                        patch_optimizer.zero_grad()

                    if step % 50 == 0 or step == refine_steps - 1:
                         logger.info(f"   [Patch Refine] Step {step}/{refine_steps}: Loss {(loss * grad_acc_steps).item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

                # ====================================================================
                # 【终极优化点】使用安全解包机制进行 SVD 吸收，彻底规避 4 和 64 碰撞！
                # ====================================================================
                logger.info("🔄 SVD Merging learned Patches into main LoRA weights before saving...")

                main_lora_alpha = float(os.environ.get('LORA_ALPHA', 64.0))
                main_lora_rank = float(os.environ.get('LORA_RANK', 64.0))
                main_scaling = main_lora_alpha / main_lora_rank

                with torch.no_grad():
                    for name, module in self.model.named_modules():
                        if hasattr(module, '_forward_hooks') and len(module._forward_hooks) > 0:
                            patch = list(module._forward_hooks.values())[0]
                            if isinstance(patch, ResidualPatch):
                                # 寻找对应层的 lora_A 和 lora_B
                                lora_A_name = name + ".lora_A.default.weight"
                                lora_B_name = name + ".lora_B.default.weight"

                                params_dict = dict(self.model.named_parameters())
                                # 兼容不同版本的 PEFT 命名
                                if lora_A_name not in params_dict:
                                    lora_A_name = name + ".lora_A.weight"
                                    lora_B_name = name + ".lora_B.weight"

                                if lora_A_name in params_dict and lora_B_name in params_dict:
                                    L_A = params_dict[lora_A_name]
                                    L_B = params_dict[lora_B_name]

                                    # 【核心修复】：使用 .data 抽取纯净张量进行矩阵运算，切断与 DeepSpeed 环境的隐式联系
                                    clean_LA = L_A.data.float()
                                    clean_LB = L_B.data.float()
                                    clean_PA = patch.patch_A.data.float()
                                    clean_PB = patch.patch_B.data.float()

                                    # 计算合并矩阵
                                    delta_W = (clean_LB @ clean_LA) * main_scaling + (clean_PB @ clean_PA) * patch.scaling
                                    target_M = delta_W / main_scaling

                                    # SVD 分解
                                    U, S, Vh = torch.linalg.svd(target_M, full_matrices=False)

                                    # 截断回主 rank 64
                                    rank = L_A.shape[0]
                                    U_trunc = U[:, :rank]
                                    S_trunc = S[:rank]
                                    Vh_trunc = Vh[:rank, :]

                                    # 生成纯净的新矩阵
                                    new_L_B = U_trunc * torch.sqrt(S_trunc)
                                    new_L_A = torch.sqrt(S_trunc).unsqueeze(1) * Vh_trunc

                                    # 强制安全覆写
                                    L_B.data.copy_(new_L_B.to(L_B.dtype))
                                    L_A.data.copy_(new_L_A.to(L_A.dtype))

                                    # 清除 Hook，确保保存环境干净
                                    module._forward_hooks.clear()

                self.save_model(output_dir=os.path.join(run_dir, "refined_final"))
                logger.info("✅ Targeted Refinement SVD Merged and Saved.")

        except Exception as e:
            logger.warning(f"⚠️ Refinement skipped: {e}")
        # ============================================================

        return TrainOutput(self.state.global_step, train_loss, metrics)







    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)


        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps



    def comput_custom_reg(self, model, reg_lambda=0.1):
        reg_loss = 0.0
        param_count = 0

        # ==================== 1. 收集参数 ====================
        dict_A, dict_B, dict_PT = {}, {}, {}
        for name, param in model.named_parameters():
            # 筛选 Lora A
            if "lora_A" in name and any(proj in name for proj in ("q_proj", "k_proj", "v_proj", "mm_projector")):
                dict_A[name] = param
            # 筛选 Lora B
            elif "lora_B" in name and any(proj in name for proj in ("q_proj", "k_proj", "v_proj", "mm_projector")):
                dict_B[name] = param
            # 筛选 Base Layer (原始权重)
            elif "base_layer" in name and any(proj in name for proj in ("q_proj", "k_proj", "v_proj", "mm_projector")):
                dict_PT[name] = param

        # ==================== 2. 计算正则化 ====================
        for lora_A_name, A in dict_A.items():
            # 匹配名称
            lora_B_name = lora_A_name.replace("lora_A", "lora_B")
            if "default" in lora_A_name:
                PT_name = lora_A_name.replace("lora_A.default", "base_layer")
            else:
                PT_name = lora_A_name.replace("lora_A", "base_layer")

            if lora_B_name not in dict_B or PT_name not in dict_PT:
                continue

            B = dict_B[lora_B_name]
            W = dict_PT[PT_name]

            with torch.no_grad():
                # ====== 【完美契合：通过环境变量直接获取主 LoRA 的 scaling】 ======
                main_lora_alpha = float(os.environ.get('LORA_ALPHA', 64.0))
                main_lora_rank = float(os.environ.get('LORA_RANK', 64.0))
                scaling = main_lora_alpha / main_lora_rank
                # ================================================================

                delta_W = (B @ A) * scaling

                # 动态有效权重
                W_effective = W + delta_W.to(W.dtype)

                # 对加上了 LoRA 的有效权重提取 HOG 边缘！
                hog_prior = compute_hog_prior(W_effective)

                # 这样 M (Mask) 就会随着训练动态演变，不仅保护原模型的边缘，
                # 还会保护 LoRA 已经学出来的关键特征边缘，防止灾难性遗忘！

                # C. Min-Max 归一化
                hog_min = hog_prior.min()
                hog_max = hog_prior.max()
                hog_norm = (hog_prior - hog_min) / (hog_max - hog_min + 1e-8)
                del hog_prior # 【关键】阅后即焚，立刻释放内存！

                # D. 计算原始的幅度 Mask
                # 为了防除零溢出，算 norm 时局部转一下，算出一个标量，不占大显存
                w_norm = torch.norm(W.to(torch.float32), p=2).to(W.dtype)
                M_pre = W / (w_norm + 1e-8)

                S_mag = torch.abs((1 / torch.log(M_pre.abs() + 1e-15)))
                del M_pre # 【关键】阅后即焚！

                # E. 融合两种先验
                S_combined = S_mag + HOG_LAMBDA * hog_norm
                del S_mag, hog_norm # 【关键】阅后即焚！

                # 生成最终 Mask
                M = torch.tanh(OMEGA * S_combined)
                del S_combined # 【关键】阅后即焚！

            # F. 计算损失 (代码不变)
            reg_loss += torch.norm(M * (B @ A), p=2)
            param_count += 1

        if param_count > 0:
            reg_loss = reg_lambda * reg_loss / param_count

        return reg_loss



    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"]

        reg_loss = self.comput_custom_reg(model, reg_lambda=CMR_LAMBDA)
        loss += reg_loss

        return (loss, outputs) if return_outputs else loss
