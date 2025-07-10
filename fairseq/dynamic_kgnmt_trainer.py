# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

import contextlib
import logging
import os
import sys
import time
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

from fairseq import checkpoint_utils, models, optim, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics
from fairseq.models.ema import build_ema
from fairseq.nan_detector import NanDetector
from fairseq.optim import lr_scheduler
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.utils import safe_hasattr

logger = logging.getLogger(__name__)

class DynamicKgNMTTrainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, cfg: FairseqConfig, task, model, criterion, quantizer=None):
        if isinstance(cfg, Namespace):
            logger.warning(
                "argparse.Namespace configuration is deprecated! Automatically converting to OmegaConf"
            )
            cfg = convert_namespace_to_omegaconf(cfg)

        self.cfg = cfg
        self.task = task

        # catalog shared parameters
        shared_params = _catalog_shared_params(model)
        self.tpu = cfg.common.tpu
        self.cuda = torch.cuda.is_available() and not cfg.common.cpu and not self.tpu
        if self.cuda:
            self.device = torch.device("cuda")
        elif self.tpu:
            self.device = utils.get_tpu_device()
        else:
            self.device = torch.device("cpu")

        if self.is_fsdp:
            import fairscale

            if self.cfg.common.bf16:
                raise ValueError(
                    "FullyShardedDataParallel is not compatible with --bf16 or "
                    "--memory-efficient-bf16"
                )
            if self.cfg.distributed_training.zero_sharding != "none":
                raise ValueError(
                    "FullyShardedDataParallel is not compatible with --zero-sharding "
                    "option (it's already built in)"
                )
            if (
                max(self.cfg.optimization.update_freq) > 1
                and fairscale.__version__ < "0.4.0"
            ):
                raise RuntimeError(
                    "Please update to fairscale 0.4.0 or newer when combining "
                    "--update-freq with FullyShardedDataParallel"
                )
        else:
            if (
                hasattr(self.cfg.distributed_training, "cpu_offload")
                and self.cfg.distributed_training.cpu_offload
            ):
                raise ValueError("--cpu-offload requires --ddp-backend=fully_sharded")

        # copy model and criterion to current device/dtype
        self._criterion = criterion
        self._model = model
        if not self.is_fsdp:
            if cfg.common.fp16:
                assert not cfg.common.amp, "Cannot use fp16 and AMP together"
                self._criterion = self._criterion.half()
                self._model = self._model.half()
            elif cfg.common.bf16:
                self._criterion = self._criterion.to(dtype=torch.bfloat16)
                self._model = self._model.to(dtype=torch.bfloat16)
            elif cfg.common.amp:
                self._amp_retries = 0
        if (
            not cfg.distributed_training.pipeline_model_parallel
            # the DistributedFairseqModel wrapper will handle moving to device,
            # so only handle cases which don't use the wrapper
            and not self.use_distributed_wrapper
        ):
            self._criterion = self._criterion.to(device=self.device)
            self._model = self._model.to(device=self.device)
        self.pipeline_model_parallel = cfg.distributed_training.pipeline_model_parallel
        self.last_device = None
        if self.cuda and self.pipeline_model_parallel:
            self.last_device = torch.device(
                cfg.distributed_training.pipeline_devices[-1]
            )

        # check that shared parameters are preserved after device transfer
        for shared_param in shared_params:
            ref = _get_module_by_path(self._model, shared_param[0])
            for path in shared_param[1:]:
                logger.info(
                    "detected shared parameter: {} <- {}".format(shared_param[0], path)
                )
                _set_module_by_path(self._model, path, ref)

        self._dummy_batch = None  # indicates we don't have a dummy batch at first
        self._num_updates = 0
        self._num_xla_compiles = 0  # for TPUs
        self._knowledge_selector_optim_history = None
        self._kgnmt_optim_history = None
        # self._optimizer = None
        self._knowledge_selector_optimizer = None
        self._kgnmt_optimizer = None
        self._knowledge_selector_lr_scheduler = None
        self._kgnmt_lr_scheduler = None
        self._warn_once = set()
        self._wrapped_criterion = None
        self._wrapped_model = None
        self._ema = None

        # TODO(myleott): support tpu
        if self.cuda and self.data_parallel_world_size > 1:
            self._grad_norm_buf = torch.cuda.DoubleTensor(self.data_parallel_world_size)
        else:
            self._grad_norm_buf = None

        self.quantizer = quantizer
        if self.quantizer is not None:
            self.quantizer.set_trainer(self)

        # get detailed cuda environment
        if self.cuda:
            self.cuda_env = utils.CudaEnvironment()
            if self.data_parallel_world_size > 1:
                self.cuda_env_arr = distributed_utils.all_gather_list(
                    self.cuda_env, group=distributed_utils.get_global_group()
                )
            else:
                self.cuda_env_arr = [self.cuda_env]
            if self.data_parallel_rank == 0:
                utils.CudaEnvironment.pretty_print_cuda_env_list(self.cuda_env_arr)
        else:
            self.cuda_env = None
            self.cuda_env_arr = None

        metrics.log_start_time("wall", priority=790, round=0)

        self._start_time = time.time()
        self._previous_training_time = 0
        self._cumulative_training_time = None

    def reinitialize(self):
        """Reinitialize the Trainer, typically after model params change."""
        self._knowledge_selector_optimizer = None
        self._kgnmt_optimizer = None
        self._knowledge_selector_lr_scheduler = None
        self._kgnmt_lr_scheduler = None
        self._wrapped_criterion = None
        self._wrapped_model = None

    @property
    def data_parallel_world_size(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()

    @property
    def data_parallel_rank(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    @property
    def is_data_parallel_master(self):
        # NOTE: this returns true for all model parallel replicas with data
        # parallel rank 0
        return self.data_parallel_rank == 0

    @property
    def use_distributed_wrapper(self) -> bool:
        return (
            self.data_parallel_world_size > 1 and not self.cfg.optimization.use_bmuf
        ) or (self.is_fsdp and self.cfg.distributed_training.cpu_offload)

    @property
    def should_save_checkpoint_on_current_rank(self) -> bool:
        """Indicates whether to save checkpoints on the current DDP rank."""
        if (
            self.is_fsdp and self.cfg.distributed_training.use_sharded_state
        ) or getattr(self.cfg.model, "base_layers", 0) > 0:
            return True
        else:
            return self.is_data_parallel_master

    @property
    def always_call_state_dict_during_save_checkpoint(self) -> bool:
        if self.is_fsdp and not self.cfg.distributed_training.use_sharded_state:
            # FSDP calls communication collective when consolidating checkpoints
            return True
        else:
            return False

    @property
    def checkpoint_suffix(self) -> str:
        """Suffix to add to the checkpoint file name."""
        if self.is_fsdp and self.cfg.distributed_training.use_sharded_state:
            return self.cfg.checkpoint.checkpoint_suffix + "-shard{0}".format(
                self.data_parallel_rank
            )
        else:
            return self.cfg.checkpoint.checkpoint_suffix or ""

    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            if utils.has_parameters(self._criterion) and self.use_distributed_wrapper:
                self._wrapped_criterion = models.DistributedFairseqModel(
                    self.cfg.distributed_training,
                    self._criterion,
                    process_group=self.data_parallel_process_group,
                    device=self.device,
                )
            else:
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.use_distributed_wrapper:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.cfg.distributed_training,
                    self._model,
                    process_group=self.data_parallel_process_group,
                    device=self.device,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def ema(self):
        if self._ema is None:
            self._build_ema()
        return self._ema

    def _build_ema(self):
        if self.cfg.ema.store_ema:
            self._ema = build_ema(self._model, self.cfg.ema, self.device)
            logger.info("Exponential Moving Average Shadow Model is initialized.")

    @property
    def knowledge_selector_optimizer(self):
        if self._knowledge_selector_optimizer is None:
            self._build_knowledge_selector_optimizer()
        return self._knowledge_selector_optimizer

    @property
    def kgnmt_optimizer(self):
        if self._kgnmt_optimizer is None:
            self._build_kgnmt_optimizer()
        return self._kgnmt_optimizer

    @property
    def knowledge_selector_lr_scheduler(self):
        if self._knowledge_selector_lr_scheduler is None:
            self._build_knowledge_selector_lr_scheduler()
        return self._knowledge_selector_lr_scheduler

    @property
    def kgnmt_lr_scheduler(self):
        if self._kgnmt_lr_scheduler is None:
            self._build_kgnmt_lr_scheduler()
        return self._kgnmt_lr_scheduler
    
# {'_name': 'adam', 'adam_betas': '(0.9, 0.98)', 'adam_eps': 1e-08, 'weight_decay': 0.0, 'use_old_adam': False, 'fp16_adam_stats': False, 'tpu': False, 'lr': [0.0005]}

    # TODO_THESIS: modify the optimizer config input
    def _build_knowledge_selector_optimizer(self):
        """Build optimizer for knowledge selector component."""
        params = list(filter(
            lambda p: p.requires_grad,
            chain(self.model.knowledge_selector.parameters())
        ))

        ks_optim_cfg = {
            "_name": self.cfg.model.knowledge_selector_optimizer,
            "adam_betas": self.cfg.model.knowledge_selector_adam_betas, 
            "adam_eps": 1.0e-08,
            "weight_decay": self.cfg.model.knowledge_selector_weight_decay,
            "use_old_adam": False,
            "fp16_adam_stats": False,
            "tpu": False,
            "lr": [self.cfg.model.knowledge_selector_lr],
        }
        ks_optim_cfg = OmegaConf.create(ks_optim_cfg)
        self._knowledge_selector_optimizer = optim.build_optimizer(ks_optim_cfg, params)

    def _build_kgnmt_optimizer(self):
        """Build optimizer for KG-NMT component."""
        params = list(filter(
            lambda p: p.requires_grad,
            chain(self.model.kgnmt.parameters(), self.criterion.parameters())
        ))

        kgnmt_optim_cfg = {
            "_name": self.cfg.model.kgnmt_optimizer,
            "adam_betas": self.cfg.model.kgnmt_adam_betas, 
            "adam_eps": 1.0e-08,
            "weight_decay": self.cfg.model.kgnmt_weight_decay,
            "use_old_adam": False,
            "fp16_adam_stats": False,
            "tpu": False,
            "lr": [self.cfg.model.kgnmt_lr],
        }
        kgnmt_optim_cfg = OmegaConf.create(kgnmt_optim_cfg)

        if self.cfg.common.fp16 or self.cfg.common.bf16 or self.cfg.common.amp:
            if self.cfg.common.memory_efficient_fp16 or self.cfg.common.memory_efficient_bf16:
                self._kgnmt_optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                    kgnmt_optim_cfg, params
                )
            elif self.cfg.common.amp:
                self._kgnmt_optimizer = optim.AMPOptimizer.build_optimizer(kgnmt_optim_cfg, params)
            else:
                self._kgnmt_optimizer = optim.FP16Optimizer.build_optimizer(kgnmt_optim_cfg, params)
        else:
            self._kgnmt_optimizer = optim.build_optimizer(kgnmt_optim_cfg, params)
        
    # {'_name': 'inverse_sqrt', 'warmup_updates': 4000, 'warmup_init_lr': -1.0, 'lr': [0.0005]}
    def _build_knowledge_selector_lr_scheduler(self):
        """Build LR scheduler for knowledge selector."""
        ks_lr_shuduler_cfg = {
            "_name": self.cfg.model.knowledge_selector_lr_scheduler,
            "warmup_updates": self.cfg.model.knowledge_selector_warmup_updates,
            "total_num_update": self.cfg.model.max_update,
            "lr": [self.cfg.model.knowledge_selector_lr]
        }
        ks_lr_shuduler_cfg = OmegaConf.create(ks_lr_shuduler_cfg)

        self._knowledge_selector_lr_scheduler = lr_scheduler.build_lr_scheduler(
            ks_lr_shuduler_cfg,
            self.knowledge_selector_optimizer,
        )
        self._knowledge_selector_lr_scheduler.step_update(0)

    def _build_kgnmt_lr_scheduler(self):
        """Build LR scheduler for KG-NMT."""
        kgnmt_lr_shuduler_cfg = {
            "_name": self.cfg.model.kgnmt_lr_scheduler,
            "warmup_updates": self.cfg.model.kgnmt_warmup_updates,
            "warmup_init_lr": self.cfg.model.kgnmt_warmup_init_lr,
            "lr": [self.cfg.model.kgnmt_lr]
        }
        kgnmt_lr_shuduler_cfg = OmegaConf.create(kgnmt_lr_shuduler_cfg)
        
        self._kgnmt_lr_scheduler = lr_scheduler.build_lr_scheduler(
            kgnmt_lr_shuduler_cfg,
            self.kgnmt_optimizer,
        )
        self._kgnmt_lr_scheduler.step_update(0)

    def _build_optimizer(self):
        if (
            self.cfg.optimization.debug_param_names
            and self.cfg.common.fp16_no_flatten_grads
        ):
            params = []
            self.param_names = []

            for n, p in chain(
                self.model.named_parameters(), self.criterion.named_parameters()
            ):
                if p.requires_grad:
                    params.append(p)
                    self.param_names.append(n)
        else:
            params = list(
                filter(
                    lambda p: p.requires_grad,
                    chain(self.model.parameters(), self.criterion.parameters()),
                )
            )

        if self.is_fsdp and self.cfg.common.fp16:
            # FullyShardedDataParallel always uses MemoryEfficientFP16 wrapper,
            # mostly for the grad scaling. But if we don't have the
            # --memory-efficient-fp16 flag set, then we're effectively doing
            # regular --fp16 and can allow the use of optimizers that would
            # otherwise be unsupported by MemoryEfficientFP16Optimizer.
            allow_unsupported = not self.cfg.common.memory_efficient_fp16
            self._kgnmt_optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                self.cfg, params, allow_unsupported=allow_unsupported
            )
        elif self.cfg.common.fp16 or self.cfg.common.bf16 or self.cfg.common.amp:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                logger.info(
                    "NOTE: your device does NOT support faster training with --fp16 or --amp, "
                    "please switch to FP32 which is likely to be faster"
                )
            if (
                self.cfg.common.memory_efficient_fp16
                or self.cfg.common.memory_efficient_bf16
            ):
                self._kgnmt_optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                    self.cfg, params
                )
            elif self.cfg.common.amp:
                self._kgnmt_optimizer = optim.AMPOptimizer.build_optimizer(self.cfg, params)
            else:
                self._kgnmt_optimizer = optim.FP16Optimizer.build_optimizer(self.cfg, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                logger.info(
                    "NOTE: your device may support faster training with --fp16 or --amp"
                )
            self._kgnmt_optimizer = optim.build_optimizer(self.cfg.optimizer, params)

        if self.is_fsdp:
            assert (
                not self.cfg.optimization.use_bmuf
            ), "--ddp-backend=fully_sharded is not compatible with BMUF"
            assert self._kgnmt_optimizer.supports_flat_params, (
                "--ddp-backend=fully_sharded is only compatible with pointwise "
                "optimizers (e.g., Adam, AdamW, Adadelta, Adamax, SGD, etc.). "
                "However, the sharding will result in slightly different results when "
                "using non-pointwise optimizers (e.g., Adagrad, Adafactor, LAMB)"
            )

        if self.cfg.optimization.use_bmuf:
            self._kgnmt_optimizer = optim.FairseqBMUF(
                self.cfg.bmuf,
                self._kgnmt_optimizer,
            )

        if self.cfg.distributed_training.zero_sharding == "os":
            if (
                self.cfg.common.fp16
                and not self.cfg.common.memory_efficient_fp16
                and not self.cfg.common.memory_efficient_bf16
            ) and not self.cfg.common.fp16_no_flatten_grads:
                raise ValueError(
                    "ZeRO is incomptabile with fp16 and flattened grads. "
                    "Please use --fp16-no-flatten-grads"
                )
            else:
                optim.shard_(self._kgnmt_optimizer, self.data_parallel_process_group)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(
            self.cfg.lr_scheduler,
            self.kgnmt_optimizer,
        )
        self._lr_scheduler.step_update(0)

    @property
    def is_fsdp(self):
        return self.cfg.distributed_training.ddp_backend == "fully_sharded"

    def consolidate_optimizer(self):
        """For OSS, we need to consolidate the state dict."""
        if self.cfg.checkpoint.no_save_optimizer_state:
            return
        self._gathered_optim_state = None
        if hasattr(self.kgnmt_optimizer.optimizer, "consolidate_state_dict"):
            self.knowledge_selector_optimizer.consolidate_state_dict()
            self.kgnmt_optimizer.optimizer.consolidate_state_dict()
        elif self.is_fsdp and not self.model.use_sharded_state:
            knowledge_selector_st = self.model.gather_full_optim_state_dict(
                self.knowledge_selector_optimizer
            )  # only returns on rank 0
            kgnmt_st = self.model.gather_full_optim_state_dict(
                self.kgnmt_optimizer
            )  # only returns on rank 0
            self._gathered_optim_state = {
                "knowledge_selector": knowledge_selector_st,
                "kgnmt": kgnmt_st,
            }

    def state_dict(self):
        state_dict = {
            "args": None,  # legacy
            "cfg": (
                OmegaConf.to_container(self.cfg, resolve=True, enum_to_str=True)
                if OmegaConf.is_config(self.cfg)
                else self.cfg
            ),
            "model": self.model.state_dict(),
            "criterion": (
                self.criterion.state_dict()
                if utils.has_parameters(self.criterion)
                else None
            ),
            "knowledge_selector_optimizer_history": (self._knowledge_selector_optim_history or [])
            + [
                {
                    # "criterion_name": self.get_criterion().__class__.__name__,
                    "optimizer_name": self.knowledge_selector_optimizer.__class__.__name__,
                    "lr_scheduler_state": self.knowledge_selector_lr_scheduler.state_dict(),
                    "num_updates": self.get_num_updates(),
                }
            ],
            "kgnmt_optimizer_history": (self._kgnmt_optim_history or [])
            + [
                {
                    "criterion_name": self.get_criterion().__class__.__name__,
                    "optimizer_name": self.kgnmt_optimizer.__class__.__name__,
                    "lr_scheduler_state": self.kgnmt_lr_scheduler.state_dict(),
                    "num_updates": self.get_num_updates(),
                }
            ],
            "task_state": self.task.state_dict() if self.task is not None else {},
            "extra_state": {
                "metrics": metrics.state_dict(),
                "previous_training_time": self.cumulative_training_time(),
            },
        }
        if self.cfg.ema.store_ema:
            # Save EMA model state as extra state
            state_dict["extra_state"]["ema"] = self.ema.get_model().state_dict()
            if self.cfg.ema.ema_fp32:
                # Save EMA params in fp32
                state_dict["extra_state"]["ema_fp32_params"] = self.ema.fp32_params
        if not self.cfg.checkpoint.no_save_optimizer_state:
            if self._gathered_optim_state is not None:
                state_dict["last_knowledge_selector_optimizer_state"] = self._gathered_optim_state["knowledge_selector"]
                state_dict["last_kgnmt_optimizer_state"] = self._gathered_optim_state["kgnmt"]
                self._gathered_optim_state = None
            else:
                state_dict["last_knowledge_selector_optimizer_state"] = self.knowledge_selector_optimizer.state_dict()
                state_dict["last_kgnmt_optimizer_state"] = self.kgnmt_optimizer.optimizer.state_dict()
        if self.is_fsdp:
            # save meta data for recombining checkpoint upon loading
            state_dict["fsdp_metadata"] = self.model.local_metadata_dict()
        return state_dict

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if self.should_save_checkpoint_on_current_rank:

            logger.info(f"Saving checkpoint to {os.path.abspath(filename)}")
            # call state_dict on all ranks in case it needs internal communication
            state_dict = utils.move_to_cpu(self.state_dict())
            state_dict["extra_state"].update(extra_state)

            checkpoint_utils.torch_persistent_save(
                state_dict,
                filename,
                async_write=self.cfg.checkpoint.write_checkpoints_asynchronously,
            )
            logger.info(f"Finished saving checkpoint to {os.path.abspath(filename)}")
            return os.path.abspath(filename)
        return None

    # TODO_THESIS: load checkpoint from file, modify this so that it can load the required model structure
    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """
        Load all training state from a checkpoint file.
        rank = 0 will load the checkpoint, and then broadcast it to all
        other ranks.
        """
        extra_state, self._knowledge_selector_optim_history, self._kgnmt_optim_history, last_optim_state = None, [], [], None

        logger.info(f"Preparing to load checkpoint {filename}")
        is_distributed = self.data_parallel_world_size > 1
        bexists = PathManager.isfile(filename)
        if bexists:
            load_on_all_ranks = (
                self.cfg.checkpoint.load_checkpoint_on_all_dp_ranks
                # TPUs don't support broadcast yet, so load checkpoints
                # on every worker for now
                or self.tpu
                # FSDP requires loading checkpoint shards on all ranks
                or (self.is_fsdp and self.cfg.distributed_training.use_sharded_state)
                or getattr(self.cfg.model, "base_layers", 0) > 0
            )

            if load_on_all_ranks or self.data_parallel_rank == 0:
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    filename, load_on_all_ranks=load_on_all_ranks
                )
                # last_optim_state = state.get("last_optimizer_state", None)
                last_optim_state = {
                    "knowledge_selector": state.get("last_knowledge_selector_optimizer_state", None),
                    "kgnmt": state.get("last_kgnmt_optimizer_state", None),
                }

                # If doing zero_sharding, do not broadcast global optimizer
                # state. Later we will broadcast sharded states to each rank
                # to avoid memory from exploding.
                if (
                    not load_on_all_ranks
                    and self.cfg.distributed_training.zero_sharding == "os"
                    and "last_optimizer_state" in state
                    and is_distributed
                ):
                    state["last_optimizer_state"] = "SHARDED"
            else:
                last_optim_state = None
                state = None

            if is_distributed and not load_on_all_ranks:
                state = distributed_utils.broadcast_object(
                    state,
                    src_rank=0,
                    group=self.data_parallel_process_group,
                    dist_device=self.device,
                )
                if self.data_parallel_rank > 0:
                    last_optim_state = {
                        "knowledge_selector": state.get("last_knowledge_selector_optimizer_state", None),
                        "kgnmt": state.get("last_kgnmt_optimizer_state", None),
                    }

            # load model parameters
            try:
                if (
                    "kgnmt_optimizer_history" in state
                    and len(state["kgnmt_optimizer_history"]) > 0
                    and "num_updates" in state["kgnmt_optimizer_history"][-1]
                ):
                    self.model.set_num_updates(
                        state["kgnmt_optimizer_history"][-1]["num_updates"]
                    )

                # this is the code related to AdaPrune
                # In short, it removes redundant heads in multi-head attention module based on heads importance provided
                # For more info, please refer to the paper: https://openreview.net/forum?id=_CMSV7FTzGI
                # The idea of prune in mha can be summarized as
                # Fine tune model (e.g. roberta encoder) on a certain datasets with regularization
                # After the model is trained. User could use get_reserve_head_index and _adaptive_prune_heads functions to get the top X heads with most importance.
                # Then user uses the rank to prune a new roberta encoder and save the pruned ckpt manually.
                # User will fine tune the the new roberta encoder via the ckpt saved above
                # To get rid of registering different pruned version of Roberta, I use the argument --mha-heads-to-keep to prune the Roberta model into a pruned version which matches the pruned ckpt.
                if (
                    safe_hasattr(self.model, "args")
                    and safe_hasattr(self.model.args, "mha_heads_to_keep")
                    and self.model.args.mha_heads_to_keep != -1
                ):
                    logger.info(
                        f"Prune model: keep {self.model.args.mha_heads_to_keep} heads for each multihead attention module"
                    )
                    for layer in self.model.encoder.sentence_encoder.layers:
                        reserve_head_index = layer.self_attn._get_reserve_head_index(
                            num_heads_to_keep=self.model.args.mha_heads_to_keep
                        )
                        layer.self_attn._adaptive_prune_heads(
                            reserve_head_index=reserve_head_index
                        )
                        layer.self_attn._set_skip_embed_dim_check()
                    logger.info(self.model)
                # this is the code related to AdaPrune
                # In short, it removes redundant units in feedforward layer in each transformer layer based on importance
                # For more info, please refer to the paper: https://openreview.net/forum?id=_CMSV7FTzGI
                # The idea of prune in ffn can be summarized as
                # Fine tune model (e.g. roberta encoder) on a certain datasets with regularization
                # After the model is trained. User could use _get_fc_rank and _prune_fc_layer functions to get the top X units with most importance.
                # Then user uses the rank to prune a new roberta encoder and save the pruned ckpt manually.
                # User will fine tune the the new roberta encoder via the ckpt saved above
                # To get rid of registering different pruned version of Roberta, I use the argument --ffn-blocks-to-remove to prune the Roberta model into a pruned version which matches the pruned ckpt.
                if (
                    safe_hasattr(self.model, "args")
                    and safe_hasattr(self.model.args, "ffn_blocks_to_remove")
                    and self.model.args.ffn_blocks_to_remove != -1
                ):
                    logger.info(
                        f"Prune model: remove {self.model.args.ffn_blocks_to_remove} ffn blocks for each transformer layer"
                    )
                    for layer in self.model.encoder.sentence_encoder.layers:
                        remove_index = layer._get_fc_rank(
                            remove_num=self.model.args.ffn_blocks_to_remove
                        )
                        layer._prune_fc_layer(remove_index=remove_index)
                    logger.info(self.model)

                self.model.load_state_dict(
                    state["model"], strict=True, model_cfg=self.cfg.model
                )
                # save memory for later steps
                del state["model"]
                if utils.has_parameters(self.get_criterion()):
                    self.get_criterion().load_state_dict(
                        state["criterion"], strict=True
                    )
                    del state["criterion"]

            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )
            extra_state = state["extra_state"]
            self._knowledge_selector_optim_history = state["knowledge_selector_optimizer_history"]
            self._kgnmt_optim_history = state["kgnmt_optimizer_history"]

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            # self._build_optimizer()
            self._build_knowledge_selector_optimizer()
            self._build_knowledge_selector_lr_scheduler()
            self._build_kgnmt_optimizer()
            self._build_kgnmt_lr_scheduler()

            # only reload optimizer and lr_scheduler if they match
            last_knowledge_selector_optim = self._knowledge_selector_optim_history[-1]
            last_kgnmt_optim = self._kgnmt_optim_history[-1]
            # TODO_THESIS: maybe we have to add another assertion here
            assert (
                last_kgnmt_optim["criterion_name"] == self.get_criterion().__class__.__name__
            ), f"Criterion does not match; please reset the optimizer (--reset-optimizer). {last_kgnmt_optim['criterion_name']} vs {self.get_criterion().__class__.__name__}"
            assert (
                last_kgnmt_optim["optimizer_name"] == self.kgnmt_optimizer.__class__.__name__
            ), f"Optimizer does not match; please reset the optimizer (--reset-optimizer). {last_kgnmt_optim['optimizer_name']} vs {self.kgnmt_optimizer.__class__.__name__}"

            if not reset_lr_scheduler:
                self.knowledge_selector_lr_scheduler.load_state_dict(
                    last_knowledge_selector_optim["lr_scheduler_state"]
                )
                self.kgnmt_lr_scheduler.load_state_dict(last_kgnmt_optim["lr_scheduler_state"])

            # TODO_THESIS: research about these 2 if brackets
            if self.is_fsdp and not self.model.use_sharded_state:
                # if use_sharded_state, the last_optim_state is already sharded, skip this
                last_optim_state = self.model.get_shard_from_optim_state_dict(
                    last_optim_state
                )
            elif not load_on_all_ranks and is_distributed:
                last_optim_state = self.kgnmt_optimizer.broadcast_global_state_dict(
                    last_optim_state
                )

            self.knowledge_selector_optimizer.load_state_dict(last_optim_state["knowledge_selector"], optimizer_overrides)
            self.kgnmt_optimizer.load_state_dict(last_optim_state["kgnmt"], optimizer_overrides)

            self.set_num_updates(last_kgnmt_optim["num_updates"])

        if extra_state is not None:
            itr_state = extra_state["train_iterator"]
            epoch = itr_state["epoch"]

            if "previous_training_time" in extra_state:
                self._previous_training_time = extra_state["previous_training_time"]
                self._start_time = time.time()

            self.lr_step(epoch)

            if (
                itr_state.get("version", 1) >= 2
                and itr_state["iterations_in_epoch"] == 0
            ):
                # reset meters at start of epoch
                reset_meters = True

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()

            if self.cfg.ema.store_ema:
                if "ema" not in extra_state:
                    logger.warn(
                        "EMA not found in checkpoint. But store_ema is True. "
                        "EMA is re-initialized from checkpoint."
                    )
                    self.ema.restore(
                        state["model"], build_fp32_params=self.cfg.ema.ema_fp32
                    )
                else:
                    logger.info("Loading EMA from checkpoint")
                    self.ema.restore(extra_state["ema"], build_fp32_params=False)

                    if self.cfg.ema.ema_fp32:
                        if "ema_fp32_params" in extra_state:
                            logger.info("Loading EMA fp32 params from checkpoint")
                            self.ema.build_fp32_params(extra_state["ema_fp32_params"])
                        else:
                            logger.info(
                                "Building EMA fp32 params from EMA model in checkpoint"
                            )
                            self.ema.build_fp32_params()

            logger.info(
                "Loaded checkpoint {} (epoch {} @ {} updates)".format(
                    filename, epoch, self.get_num_updates()
                )
            )

        else:
            logger.info("No existing checkpoint found {}".format(filename))

        return extra_state

    # TODO_THESIS: this is how the batch iterator is created - call which will call for an iterator that can contains and loads data
    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=True,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            logger.info("loading train data for epoch {}".format(epoch))
            self.task.load_dataset(
                self.cfg.dataset.train_subset,
                epoch=epoch,
                combine=combine,
                data_selector=data_selector,
                tpu=self.tpu,
            )
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(self.cfg.dataset.train_subset),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
                self.cfg.dataset.max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=(self.cfg.common.seed + epoch)
            if self.cfg.dataset.update_ordered_indices_seed
            else self.cfg.common.seed,
            num_shards=self.data_parallel_world_size if shard_batch_itr else 1,
            shard_id=self.data_parallel_rank if shard_batch_itr else 0,
            num_workers=self.cfg.dataset.num_workers,
            epoch=epoch,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
            skip_remainder_batch=self.cfg.optimization.skip_remainder_batch,
            grouped_shuffling=self.cfg.dataset.grouped_shuffling,
            update_epoch_batch_itr=self.cfg.dataset.update_epoch_batch_itr,
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def get_valid_iterator(
        self,
        subset,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over given validation subset for a given epoch."""
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            max_tokens=self.cfg.dataset.max_tokens_valid,
            max_sentences=self.cfg.dataset.batch_size_valid,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
            ),
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.cfg.dataset.num_workers,
            # always pass a fixed "epoch" to keep validation data consistent
            # across training epochs
            epoch=1,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
            skip_remainder_batch=False,
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch."""
        logger.info("begin training epoch {}".format(epoch))

        self.lr_step_begin_epoch(epoch)

        if self.quantizer is not None:
            self.quantizer.begin_epoch(epoch)

        # task specific setup per epoch
        self.task.begin_epoch(epoch, self.get_model())

        if self.tpu:
            import torch_xla.core.xla_model as xm

            xm.rendezvous("begin_epoch")  # wait for all workers
            xm.mark_step()

    # TODO_THESIS: this maybe, load the epoch that will start validating, like a flag, not directly calculation
    def begin_valid_epoch(self, epoch):
        """Called at the beginning of each validation epoch."""

        # task specific setup per validation epoch
        self.task.begin_valid_epoch(epoch, self.get_model())

    def reset_dummy_batch(self, batch):
        self._dummy_batch = batch

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False):
        self._set_seed()
        self.criterion.train()
        self.zero_grad()
        metrics.log_start_time("train_wall", priority=800, round=0)

        extra_kwargs = {}
        if self.cfg.ema.store_ema and getattr(self.task, "uses_ema", False):
            extra_kwargs["ema_model"] = self.ema.get_model()

        has_oom = False
        logging_outputs, sample_size_total, ooms = [], 0, 0

        DEBUG_LOG_PATH = os.path.join(self.cfg.checkpoint.save_dir, "debug_kgnmt.log")
        torch.set_printoptions(threshold=100000, edgeitems=100)

        for i, sample in enumerate(samples):
            sample, is_dummy_batch = self._prepare_sample(sample)

            def maybe_no_sync():
                if (
                    self.data_parallel_world_size > 1
                    and hasattr(self.model, "no_sync")
                    and i < len(samples) - 1
                    and not self.is_fsdp
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()

            try:
                with maybe_no_sync():
                    # Phase 1: Train knowledge selector
                    ks_loss, reward, modified_sample = self._train_knowledge_selector_phase(sample, is_dummy_batch)

                    if self.cuda and self.get_num_updates() == 0:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    # Phase 2: Train KG-NMT
                    kgnmt_loss, sample_size, kgnmt_logging_output = self._train_kgnmt_phase(modified_sample, is_dummy_batch)

                    if torch.isnan(sample["target"]).any() or torch.isinf(sample["target"]).any():
                        raise ValueError("Target contains NaN/Inf")

                    if sample["target"].max().item() >= self.model.kgnmt.decoder.output_projection.out_features:
                        raise ValueError("Target contains OOV token")

                    logging_output = {
                        **kgnmt_logging_output,
                        "knw_sel_reward": reward.mean().item() if torch.isfinite(reward).all() else 0.0,
                        "knw_sel_loss": ks_loss.item() if torch.isfinite(ks_loss) else 0.0,
                        "sample_size": sample_size
                    }
                    logging_outputs.append(logging_output)
                    sample_size_total += sample_size
                    del ks_loss
                    del kgnmt_loss

                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    has_oom = True
                    if raise_oom:
                        raise e
                else:
                    raise e

            except Exception:
                self.consolidate_optimizer()
                self.save_checkpoint(os.path.join(self.cfg.checkpoint.save_dir, "crash.pt"), {})
                raise

            if has_oom:
                logger.warning("Attempting to recover from OOM")
                ooms += 1
                self.zero_grad()
                if self.cuda:
                    torch.cuda.empty_cache()
                if self.cfg.distributed_training.distributed_world_size == 1:
                    return None
                logging_outputs.append({"sample_size": 0})

        if sample_size_total == 0:
            logger.warning("sample_size_total is 0, skipping update.")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return None

        self.set_num_updates(self.get_num_updates() + 1)

        if self.cfg.ema.store_ema:
            self.ema.step(self.get_model(), self.get_num_updates())
            metrics.log_scalar("ema_decay", self.ema.get_decay(), priority=10000, round=5, weight=0)

        for opt, name in [
            (self.knowledge_selector_optimizer, "ks"),
            (self.kgnmt_optimizer, "kgnmt"),
        ]:
            if hasattr(opt, "scaler"):
                scale = opt.scaler.loss_scale if self.cfg.common.fp16 else opt.scaler.get_scale()
                metrics.log_scalar(f"loss_scale_{name}", scale, priority=700, round=4, weight=0)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        logging_output = self._reduce_and_log_stats(logging_outputs, sample_size_total)
        metrics.log_stop_time("train_wall")
        return logging_output

    def _train_knowledge_selector_phase(self, sample, is_dummy_batch):
        self.model.kgnmt.eval()
        self.model.knowledge_selector.train()

        with torch.cuda.amp.autocast(enabled=isinstance(self.knowledge_selector_optimizer, AMPOptimizer)):
            ks_output = self.model.knowledge_selector(
                sample["net_input"]["src_tokens_ks"],
                sample["net_input"]["src_lengths_ks"],
                sample["net_input"]["knw_tokens"],
                sample["net_input"]["knw_lengths"],
                sample_times=getattr(self.cfg.task, 'sample_times', 5),
            )

            with torch.no_grad():
                reward = self._compute_log_prob_of_target(self.model.kgnmt, sample)

        # Normalize and protect reward
        reward = reward.detach()
        reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)

        # Detach selected triples to avoid accidental backward
        ks_output["selected_knw_tokens"] = ks_output["selected_knw_tokens"].detach()
        ks_output["selected_knw_lengths"] = ks_output["selected_knw_lengths"].detach()

        # Compute loss with clipped advantage
        baseline = reward.mean()
        advantage = (reward - baseline).clamp(min=-5, max=5)

        log_p_t = torch.nan_to_num(ks_output["log_p_t"], nan=0.0, posinf=0.0, neginf=0.0)
        loss = -(advantage * log_p_t).mean()

        if not torch.isfinite(loss):
            print(">>> NaN/Inf in KS loss")
            raise FloatingPointError("KS loss is invalid")

        self.knowledge_selector_optimizer.backward(loss)

        if not is_dummy_batch:
            self._process_gradients(
                optimizer=self.knowledge_selector_optimizer,
                model=self.model.knowledge_selector,
                sample_size=reward.size(0),
                tag="ks"
            )
            self.knowledge_selector_optimizer.step()
            self.knowledge_selector_optimizer.zero_grad()

        # Clean up input to avoid redundant memory
        del sample["net_input"]["src_tokens_ks"]
        del sample["net_input"]["src_lengths_ks"]
        sample["net_input"]["knw_tokens"] = ks_output["selected_knw_tokens"]
        sample["net_input"]["knw_lengths"] = ks_output["selected_knw_lengths"]

        return loss, reward, sample

    def _train_kgnmt_phase(self, sample, is_dummy_batch):
        self.model.knowledge_selector.eval()
        self.model.kgnmt.train()


        if torch.isnan(sample["target"]).any() or torch.isinf(sample["target"]).any():
            raise ValueError("Target contains NaN/Inf")

        if sample["target"].max().item() >= self.model.kgnmt.decoder.output_projection.out_features:
            raise ValueError("Target contains OOV token")

        with torch.cuda.amp.autocast(enabled=isinstance(self.kgnmt_optimizer, AMPOptimizer)):
            loss, sample_size, logging_output = self.criterion(self.model.kgnmt, sample)
            if is_dummy_batch:
                loss *= 0

        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=-1e3)

        if not torch.isfinite(loss):
            raise FloatingPointError("KG-NMT loss is invalid")

        self.kgnmt_optimizer.backward(loss)

        if not is_dummy_batch:
            self._process_gradients(
                optimizer=self.kgnmt_optimizer,
                model=self.model.kgnmt,
                sample_size=sample_size,
                tag="kgnmt"
            )
            self.kgnmt_optimizer.step()
            self.kgnmt_optimizer.zero_grad()

        return loss, sample_size, logging_output

    def _process_gradients(self, optimizer, model, sample_size, tag):
        optimizer.all_reduce_grads(model)

        if sample_size == 0:
            logger.warning(f"Sample size is 0 in {tag} phase, skipping gradient step.")
            return

        numer = self.data_parallel_world_size if not self.cfg.optimization.use_bmuf else 1
        denom = sample_size if sample_size > 0 else 1.0
        optimizer.multiply_grads(numer / denom)

        grad_norm = self.clip_grad_norm(self.cfg.optimization.clip_norm, optimizer)


        if not self.tpu and not torch.isfinite(grad_norm).all():
            if self.cfg.common.amp:
                logger.warning(f"AMP Overflow in {tag}, skipping step.")
                return
            else:
                raise FloatingPointError("NaN/Inf in gradients.")

        metrics.log_scalar(f"{tag}_gnorm", grad_norm, priority=400, round=3)

    def _compute_log_prob_of_target(self, model, sample):
        exclude_keys = {"src_tokens_ks", "src_lengths_ks"}
        decoder_out = model(**{k: v for k, v in sample["net_input"].items() if k not in exclude_keys})
        lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
        target = sample["target"]
        pad_mask = target.ne(model.decoder.padding_idx)
        lprobs_for_target = lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        log_prob_per_sample = (lprobs_for_target * pad_mask).sum(dim=1)
        return log_prob_per_sample


    @metrics.aggregate("valid")
    def valid_step(self, sample, raise_oom=False):
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample, is_dummy_batch = self._prepare_sample(sample)

            try:
                # === 1. Run KnowledgeSelector to get selected knowledge tokens ===
                selector_output = self.model.knowledge_selector(
                    src_tokens=sample["net_input"]["src_tokens_ks"],
                    src_lengths=sample["net_input"]["src_lengths_ks"],
                    knw_tokens=sample["net_input"]["knw_tokens"],
                    knw_lengths=sample["net_input"]["knw_lengths"],
                    sample_times=getattr(self.cfg.task, "sample_times", 5),
                    return_all_hiddens=True
                )

                # === 2. Replace knw_tokens and knw_lengths in sample with selected ones ===
                del sample["net_input"]["src_tokens_ks"]
                del sample["net_input"]["src_lengths_ks"]
                sample["net_input"]["knw_tokens"] = selector_output["selected_knw_tokens"]
                sample["net_input"]["knw_lengths"] = selector_output["selected_knw_lengths"]

                # === 3. Forward through KgNMT only ===
                net_output = self.model.kgnmt(**sample["net_input"])

                # === 4. Compute loss ===
                loss, nll_loss = self.criterion.compute_loss(
                    self.model.kgnmt, net_output, sample, reduce=True
                )
                sample_size = (
                    sample["target"].size(0) if self.criterion.sentence_avg else sample["ntokens"]
                )
                logging_output = {
                    "loss": loss.data,
                    "nll_loss": nll_loss.data,
                    "ntokens": sample["ntokens"],
                    "nsentences": sample["target"].size(0),
                    "sample_size": sample_size,
                }

                # === 5. If BLEU is enabled, compute BLEU score ===
                if self.cfg.task.eval_bleu:
                    logging_output = self.task.bleu_valid_step(sample, self.model, logging_output)

                # === 6. If accuracy reporting is enabled, compute accuracy ===
                if getattr(self.criterion, "report_accuracy", False):
                    n_correct, total = self.criterion.compute_accuracy(
                        self.model.kgnmt, net_output, sample
                    )
                    logging_output["n_correct"] = utils.item(n_correct)
                    logging_output["total"] = utils.item(total)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if not raise_oom:
                        logger.warning("OOM during valid_step, retrying batch")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)
                raise e

            # === 7. Reduce & return ===
            logging_outputs = [logging_output]
            if is_dummy_batch:
                sample_size = 0 if not torch.is_tensor(sample_size) else sample_size.zero_()

            if self.data_parallel_world_size > 1:
                logging_outputs, (sample_size,) = self._aggregate_logging_outputs(
                    logging_outputs, sample_size, ignore=is_dummy_batch
                )

            logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)
            return logging_output

    def zero_grad(self):
        self.knowledge_selector_optimizer.zero_grad()
        self.kgnmt_optimizer.zero_grad()

    def lr_step_begin_epoch(self, epoch):
        """Adjust the learning rate at the beginning of the epoch."""
        self.knowledge_selector_lr_scheduler.step_begin_epoch(epoch)
        self.kgnmt_lr_scheduler.step_begin_epoch(epoch)
        # prefer updating the LR based on the number of steps
        ks_lr = self.lr_step_update(self.knowledge_selector_lr_scheduler)
        kgnmt_lr = self.lr_step_update(self.kgnmt_lr_scheduler)

        return ks_lr, kgnmt_lr

    # TODO_THESIS: modify this
    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.knowledge_selector_lr_scheduler.step(epoch)
        self.kgnmt_lr_scheduler.step(epoch)
        # prefer updating the LR based on the number of steps

        ks_lr = self.lr_step_update(self.knowledge_selector_lr_scheduler)
        kgnmt_lr = self.lr_step_update(self.kgnmt_lr_scheduler)

        return ks_lr, kgnmt_lr

    def lr_step_update(self, lr_scheduler):
        """Update the learning rate after each update."""
        new_lr = lr_scheduler.step_update(self.get_num_updates())
        if isinstance(new_lr, dict):
            for k, v in new_lr.items():
                metrics.log_scalar(f"lr_{k}", v, weight=0, priority=300)
            new_lr = new_lr.get("default", next(iter(new_lr.values())))
        else:
            metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def get_lr(self):
        """Get the current learning rate."""
        return (self.knowledge_selector_optimizer.get_lr(), self.kgnmt_optimizer.get_lr())

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self._criterion

    def get_meter(self, name):
        """[deprecated] Get a specific meter by name."""
        from fairseq import meters

        if "get_meter" not in self._warn_once:
            self._warn_once.add("get_meter")
            utils.deprecation_warning(
                "Trainer.get_meter is deprecated. Please use fairseq.metrics instead."
            )

        train_meters = metrics.get_meters("train")
        if train_meters is None:
            train_meters = {}

        if name == "train_loss" and "loss" in train_meters:
            return train_meters["loss"]
        elif name == "train_nll_loss":
            # support for legacy train.py, which assumed this meter is
            # always initialized
            m = train_meters.get("nll_loss", None)
            return m or meters.AverageMeter()
        elif name == "wall":
            # support for legacy train.py, which assumed this meter is
            # always initialized
            m = metrics.get_meter("default", "wall")
            return m or meters.TimeMeter()
        elif name == "wps":
            m = metrics.get_meter("train", "wps")
            return m or meters.TimeMeter()
        elif name in {"valid_loss", "valid_nll_loss"}:
            # support for legacy train.py, which assumed these meters
            # are always initialized
            k = name[len("valid_") :]
            m = metrics.get_meter("valid", k)
            return m or meters.AverageMeter()
        elif name == "oom":
            return meters.AverageMeter()
        elif name in train_meters:
            return train_meters[name]
        return None

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update(self.knowledge_selector_lr_scheduler)
        self.lr_step_update(self.kgnmt_lr_scheduler)
        if self.quantizer:
            self.quantizer.step_update(self._num_updates)
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def clip_grad_norm(self, clip_norm, optimizer):
        def agg_norm_fn(total_norm):
            total_norm = total_norm.cuda().float() ** 2
            total_norm = distributed_utils.all_reduce(
                total_norm, group=self.data_parallel_process_group
            )
            return total_norm**0.5

        should_agg_norm = self.is_fsdp and (
            self.data_parallel_process_group is not None
            or torch.distributed.is_initialized()
        )

        return optimizer.clip_grad_norm(
            clip_norm, aggregate_norm_fn=agg_norm_fn if should_agg_norm else None
        )


    def cumulative_training_time(self):
        if self._cumulative_training_time is None:
            # single GPU
            return self._local_cumulative_training_time()
        else:
            return self._cumulative_training_time

    def _local_cumulative_training_time(self):
        """Aggregate training time in seconds."""
        return time.time() - self._start_time + self._previous_training_time

    def _fp_convert_sample(self, sample):
        def apply_half(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.half)
            return t

        def apply_bfloat16(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.bfloat16)
            return t

        if self.cfg.common.fp16:
            sample = utils.apply_to_sample(apply_half, sample)

        if self.cfg.common.bf16:
            sample = utils.apply_to_sample(apply_bfloat16, sample)

        return sample

    def _prepare_sample(self, sample, is_dummy=False):
        if sample == "DUMMY":
            raise Exception(
                "Trying to use an uninitialized 'dummy' batch. This usually indicates "
                "that the total number of batches is smaller than the number of "
                "participating GPUs. Try reducing the batch size or using fewer GPUs."
            )

        if sample is None or len(sample) == 0:
            assert (
                self._dummy_batch is not None and len(self._dummy_batch) > 0
            ), "Invalid dummy batch: {}".format(self._dummy_batch)
            sample, _ = self._prepare_sample(self._dummy_batch, is_dummy=True)
            return sample, True

        # Given that PCIe/NVLink bandwidth is significantly smaller than DRAM bandwidth
        # it makes sense to do the format conversion on the CPU and then transfer
        # a smaller buffer to the device. This also saves GPU memory capacity.

        if self.cfg.common.on_cpu_convert_precision:
            sample = self._fp_convert_sample(sample)

        if self.cuda:
            if self.pipeline_model_parallel:
                if "target" in sample:
                    sample["target"] = utils.move_to_cuda(
                        sample["target"], device=self.last_device
                    )
            else:
                sample = utils.move_to_cuda(sample)
        elif self.tpu and is_dummy:
            # the dummy batch may not be on the appropriate device
            sample = utils.move_to_cuda(sample, device=self.device)

        if not self.cfg.common.on_cpu_convert_precision:
            sample = self._fp_convert_sample(sample)

        if self._dummy_batch == "DUMMY":
            self._dummy_batch = sample

        return sample, False

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.cfg.common.seed + self.get_num_updates()
        utils.set_torch_seed(seed)

    def _sync_stats(self):
        # Return True if it's using multiple GPUs and DDP or multiple GPUs with
        # BMUF and it's a bmuf sync with warmup iterations completed before.
        if self.data_parallel_world_size == 1:
            return False
        elif self.cfg.optimization.use_bmuf:
            return (
                self.get_num_updates() + 1
            ) % self.cfg.bmuf.global_sync_iter == 0 and (
                self.get_num_updates() + 1
            ) > self.cfg.bmuf.warmup_iterations
        else:
            return True

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()

    def _aggregate_logging_outputs(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        if self.task.__class__.logging_outputs_can_be_summed(self.get_criterion()):
            return self._fast_stat_sync_sum(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )
        else:
            return self._all_gather_list_sync(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        if self.tpu:
            raise NotImplementedError
        if ignore:
            logging_outputs = []
        results = list(
            zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs] + list(extra_stats_to_sum),
                    max_size=getattr(self.cfg.common, "all_gather_list_size", 16384),
                    group=self.data_parallel_process_group,
                )
            )
        )
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _fast_stat_sync_sum(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. fast_stat_sync_sum is
        faster than all_gather_list_sync, but is only suitable when
        logging outputs are scalars and can be summed. Note that
        *logging_outputs* cannot contain any nested dicts/lists.
        """
        data = {}
        for i, stat in enumerate(extra_stats_to_sum):
            data["extra_stats_" + str(i)] = stat
        if len(logging_outputs) > 0:
            log_keys = list(logging_outputs[0].keys())
            for k in log_keys:
                if not ignore:
                    v = sum(log[k] for log in logging_outputs if k in log)
                else:
                    v = logging_outputs[0][k]
                    v = torch.zeros_like(v) if torch.is_tensor(v) else 0
                data["logging_outputs_" + k] = v
        else:
            log_keys = None

        data = distributed_utils.all_reduce_dict(
            data, device=self.device, group=self.data_parallel_process_group
        )

        extra_stats_to_sum = [
            data["extra_stats_" + str(i)] for i in range(len(extra_stats_to_sum))
        ]
        if log_keys is not None:
            logging_outputs = [{k: data["logging_outputs_" + k] for k in log_keys}]
        else:
            logging_outputs = []
        return logging_outputs, extra_stats_to_sum

    def _check_grad_norms(self, grad_norm):
        """Check that grad norms are consistent across workers."""
        if self._grad_norm_buf is not None:
            self._grad_norm_buf.zero_()
            self._grad_norm_buf[self.data_parallel_rank] = grad_norm
            distributed_utils.all_reduce(
                self._grad_norm_buf, group=self.data_parallel_process_group
            )

            def is_consistent(tensor):
                max_abs_diff = torch.max(torch.abs(tensor - tensor[0]))
                return (
                    (
                        torch.isfinite(tensor).all()
                        and (max_abs_diff / (tensor[0] + 1e-6) < 1e-6).all()
                    )
                    or (self.cfg.common.amp and not torch.isfinite(tensor).all())
                    # in case of amp non-finite grads are fine
                )

            if not is_consistent(self._grad_norm_buf):
                pretty_detail = "\n".join(
                    "rank {:3d} = {:.8f}".format(r, n)
                    for r, n in enumerate(self._grad_norm_buf.tolist())
                )
                error_detail = "grad_norm across the workers:\n{}\n".format(
                    pretty_detail
                )
                # use FloatingPointError to trigger NanDetector
                raise FloatingPointError(
                    "Fatal error: gradients are inconsistent between workers. "
                    "Try --ddp-backend=legacy_ddp. "
                    "Or are you mixing up different generation of GPUs in training?"
                    + "\n"
                    + "-" * 80
                    + "\n{}\n".format(error_detail)
                    + "-" * 80
                )

    def _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm=None):
        if grad_norm is not None and (
            not torch.is_tensor(grad_norm) or torch.isfinite(grad_norm)
        ):
            metrics.log_speed("ups", 1.0, priority=100, round=2)
            metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
            if self.cfg.optimization.clip_norm > 0:
                metrics.log_scalar(
                    "clip",
                    torch.where(
                        grad_norm > self.cfg.optimization.clip_norm,
                        grad_norm.new_tensor(100),
                        grad_norm.new_tensor(0),
                    ),
                    priority=500,
                    round=1,
                )

        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.task.reduce_metrics(logging_outputs, self.get_criterion())
                del logging_outputs

            # extra warning for criterions that don't properly log a loss value
            if "loss" not in agg:
                if "loss" not in self._warn_once:
                    self._warn_once.add("loss")
                    logger.warning(
                        "Criterion.reduce_metrics did not log a 'loss' value, "
                        "which may break some functionality"
                    )
                metrics.log_scalar("loss", -1)

            # support legacy interface
            if self.tpu:
                logging_output = {}
            else:
                logging_output = agg.get_smoothed_values()
                logging_output["sample_size"] = sample_size
                for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                    if key_to_delete in logging_output:
                        del logging_output[key_to_delete]
            return logging_output

    def _check_xla_compilation(self):
        import torch_xla.debug.metrics as met

        compile_stats = met.metric_data("CompileTime")
        if compile_stats is None:
            return
        num_xla_compiles = compile_stats[0]
        if num_xla_compiles > self._num_xla_compiles:
            logger.warning(
                "XLA compilation detected on device #{}; too many of these can lead "
                "to slow training, but we expect a few in the beginning".format(
                    self.cfg.distributed_training.distributed_rank
                )
            )
        self._num_xla_compiles = num_xla_compiles

    def _xla_markstep_and_send_to_cpu(self, data=None):
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        if data is not None:
            from fairseq.utils import xla_device_to_cpu

            return xla_device_to_cpu(data)


def _catalog_shared_params(module, memo=None, prefix=""):
    if memo is None:
        first_call = True
        memo = {}
    else:
        first_call = False
    for name, param in module._parameters.items():
        param_prefix = prefix + ("." if prefix else "") + name
        if param not in memo:
            memo[param] = []
        memo[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        _catalog_shared_params(m, memo, submodule_prefix)
    if first_call:
        return [x for x in memo.values() if len(x) > 1]


def _get_module_by_path(module, path):
    path = path.split(".")
    for name in path:
        module = getattr(module, name)
    return module


def _set_module_by_path(module, path, value):
    path = path.split(".")
    for name in path[:-1]:
        module = getattr(module, name)
    setattr(module, path[-1], value)
