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
    KgNMTConfig,
    KgNMTDecoderBase,
    KgNMTEncoderBase,
    KgNMTKnowledgeEncoderBase,
)

logger = logging.getLogger(__name__)

def check_type(module, expected_type):
    if hasattr(module, "unwrapped_module"):
        assert isinstance(
            module.unwrapped_module, expected_type
        ), f"{type(module.unwrapped_module)} != {expected_type}"
    else:
        assert isinstance(module, expected_type), f"{type(module)} != {expected_type}"

class KgNMTModelBase(BaseFairseqModel):
    """
    Base model for KgNMT, core model of Dynamic-KgNMT

    Args:
        knw_encoder (KgNMTKnowledgeEncoder): the knowledge encoder
        encoder (KgNMTEncoder): the source encoder
        decoder (KgNMTDecoder): the decoder

    The KgNMT model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.dynamic_kgnmt_parser
        :prog:
    """

    def __init__(self, cfg, encoder, knw_encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.knw_encoder = knw_encoder
        self.decoder = decoder

        check_type(self.encoder, KgNMTEncoderBase)
        check_type(self.knw_encoder, KgNMTKnowledgeEncoderBase)
        check_type(self.decoder, KgNMTDecoderBase)

        self.cfg = cfg
        self.supports_align_args = True

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, KgNMTConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, knw_dict, tgt_dict = task.source_dictionary, task.knowledge_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            knw_encoder_embed_tokens = cls.build_embedding(
                cfg, knw_dict, cfg.knw_encoder.embed_dim, cfg.knw_encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        knw_encoder = cls.build_knw_encoder(cfg, knw_dict, knw_encoder_embed_tokens) # shared embedding
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder=encoder, knw_encoder=knw_encoder, decoder=decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return KgNMTEncoderBase(cfg, src_dict, embed_tokens)
    
    @classmethod
    def build_knw_encoder(cls, cfg, knw_dict, embed_tokens):
        return KgNMTKnowledgeEncoderBase(cfg, knw_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return KgNMTDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        knw_tokens,
        knw_lengths,
        knw_z_lengths,
        prev_output_tokens,
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
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )

        knw_encoder_out = self.knw_encoder(
            knw_tokens, 
            src_lengths=knw_lengths, 
            knw_sep=True,
            return_all_hiddens=return_all_hiddens
        )

        knw_enc = knw_encoder_out["encoder_out"][0]  # (T_knw, B, C)
        knw_enc = knw_enc.transpose(0, 1)  # (B, T_knw, C)
        triple_indices = knw_encoder_out["triple_indices"][0]  # (B, T_knw)
        triple_indices = triple_indices.to(knw_enc.device)

        B, T_knw, C = knw_enc.size()
        device = knw_enc.device
        Z = triple_indices.max().item() + 1  # number of triples per example

        # === Efficient triple aggregation using index_add ===
        flat_enc = knw_enc.reshape(B * T_knw, C)          # (B*T, C)
        flat_indices = triple_indices.reshape(B * T_knw)  # (B*T,)
        valid_mask = flat_indices >= 0

        flat_enc_valid = flat_enc[valid_mask]
        flat_indices_valid = flat_indices[valid_mask]

        batch_ids = torch.arange(B, device=device).unsqueeze(1).repeat(1, T_knw).reshape(-1)
        batch_ids = batch_ids[valid_mask]
        global_indices = batch_ids * Z + flat_indices_valid  # (N_valid,)

        z_sum = torch.zeros(B * Z, C, device=device)
        z_sum.index_add_(0, global_indices, flat_enc_valid)

        flat_ones = torch.ones_like(flat_indices_valid, dtype=torch.float).unsqueeze(1)  # (N_valid, 1)
        z_count = torch.zeros(B * Z, 1, device=device)
        z_count.index_add_(0, global_indices, flat_ones)

        z_mean = z_sum / (z_count + 1e-6)
        z_mean = z_mean.reshape(B, Z, C)

        # === Create new padding mask ===
        max_len = knw_z_lengths.max().item()
        idx = torch.arange(max_len, device=knw_z_lengths.device).unsqueeze(0) # (1, T)
        pad_left = max_len - knw_z_lengths.unsqueeze(1) # (B, 1)
        mask = idx < pad_left
        
        z_mean = z_mean.transpose(0, 1)
        knw_encoder_out["encoder_out"] = [z_mean] # [Z x B x C]
        knw_encoder_out["encoder_padding_mask"] = [mask]

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            knw_encoder_out=knw_encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            knw_lengths=knw_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

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


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
