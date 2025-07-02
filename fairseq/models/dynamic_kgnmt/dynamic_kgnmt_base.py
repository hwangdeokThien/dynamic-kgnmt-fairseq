# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import logging

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import BaseFairseqModel
from fairseq.models.dynamic_kgnmt import (
    DynamicKgNMTConfig,
    KgNMTModelBase,
    KnowledgeSelectorBase,
)


logger = logging.getLogger(__name__)

def check_type(module, expected_type):
    if hasattr(module, "unwrapped_module"):
        assert isinstance(
            module.unwrapped_module, expected_type
        ), f"{type(module.unwrapped_module)} != {expected_type}"
    else:
        assert isinstance(module, expected_type), f"{type(module)} != {expected_type}"

class DynamicKgNMTModelBase(BaseFairseqModel):
    """
    Synthetic model, include Knowledge Selector and KgNMT core-model.

    Args:
        knowledge_selector (): 
        kgnmt ():

    The KgNMT model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.dynamic_kgnmt_parser
        :prog:
    """

    def __init__(self, cfg, knowledge_selector, kgnmt):
        super().__init__()
        
        # self.knowledge_selector = fsdp_wrap(knowledge_selector)
        self.knowledge_selector = knowledge_selector
        self.kgnmt = kgnmt

        check_type(self.knowledge_selector, KnowledgeSelectorBase)
        check_type(self.kgnmt, KgNMTModelBase)

        self.cfg = cfg
        self.supports_align_args = True

    # TODO_THESIS: build model function
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        if not hasattr(cfg, "kgnmt") or not hasattr(cfg, "knowledge_selector"):
            raise ValueError("DynamicKgNMTConfig must contain both `kgnmt` and `knowledge_selector` fields")

        knowledge_selector = cls.build_knowledge_selector(cfg.knowledge_selector, task)
        kgnmt = cls.build_kgnmt(cfg.kgnmt, task)

        return cls(cfg, knowledge_selector=knowledge_selector, kgnmt=kgnmt)

    @classmethod
    def build_knowledge_selector(cls, args, task):
        """Build a new knowledge selector instance."""
        return KnowledgeSelectorBase.build_model(args, task)
    
    @classmethod
    def build_kgnmt(cls, args, task):
        """Build a new KgNMT instance."""
        return KgNMTModelBase.build_model(args, task)
    
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        gen_parser_from_dataclass(
            parser,
            DynamicKgNMTConfig(),
            delete_default=False,
            with_prefix=""
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        knw_tokens,
        knw_lengths,
        prev_output_tokens,
        sample_times: Optional[int] = 5,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.
 
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        # TODO_THESIS: modify the forward

        # forward knowledge selector - get knowledge triples
        knw_selector_out = self.knowledge_selector(
            src_tokens, src_lengths, knw_tokens, knw_lengths, sample_times
        )

        selected_knw_tokens = knw_selector_out["selected_knw_tokens"]
        selected_knw_lengths = knw_selector_out["selected_knw_lengths"]

        # forward kgnmt model - compute the output
        output = self.kgnmt(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            knw_tokens=selected_knw_tokens,
            knw_lengths=selected_knw_lengths,
            prev_output_tokens=prev_output_tokens,
            return_all_hiddens=return_all_hiddens,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        return output

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

