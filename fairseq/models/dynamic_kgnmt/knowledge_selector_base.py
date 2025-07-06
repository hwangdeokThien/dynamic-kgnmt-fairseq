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
    KgNMTEncoderBase,
    KgNMTKnowledgeEncoderBase,
    Embedding
)

logger = logging.getLogger(__name__)

def check_type(module, expected_type):
    if hasattr(module, "unwrapped_module"):
        assert isinstance(
            module.unwrapped_module, expected_type
        ), f"{type(module.unwrapped_module)} != {expected_type}"
    else:
        assert isinstance(module, expected_type), f"{type(module)} != {expected_type}"

class KnowledgeSelectorBase(BaseFairseqModel):
    """
    Knowledge selector model for Dynamic-KgNMT
    Dual encoder model that combines a knowledge encoder and a source encoder.

    Args:
        knw_encoder (KgNMTEncoder): the knowledge encoder
        encoder (KgNMTEncoder): the source encoder

    The KgNMT model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.kgnmt_parser
        :prog:
    """

    def __init__(self, cfg, encoder, knw_encoder):
        super().__init__()
        
        self.encoder = encoder
        self.knw_encoder = knw_encoder
        # self.knw_extractor = knw_extractor

        check_type(self.encoder, KgNMTEncoderBase)
        check_type(self.knw_encoder, KgNMTKnowledgeEncoderBase)

        self.cfg = cfg

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
        
        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))

        src_dict, knw_dict = task.source_dictionary, task.knowledge_dictionary

        encoder_embed_tokens = cls.build_embedding(
            cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
        )
        knw_encoder_embed_tokens = cls.build_embedding(
            cfg, knw_dict, cfg.knw_encoder.embed_dim, cfg.knw_encoder.embed_path
        )

        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing

        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        knw_encoder = cls.build_knw_encoder(cfg, knw_dict, knw_encoder_embed_tokens)

        return cls(cfg, encoder=encoder, knw_encoder=knw_encoder)

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
    def build_knw_encoder(cls, cfg, src_dict, embed_tokens):
        return KgNMTKnowledgeEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_knw_extractor(cls, cfg, src_dict, embed_tokens):
        """Build a new knowledge extractor instance."""
        # This is a placeholder for the knowledge extractor, which can be implemented later.
        # For now, we return None or a dummy extractor.
        return None

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        knw_tokens,
        knw_lengths,
        sample_times: Optional[int] = 5,
        return_all_hiddens: bool = True,
    ):
        """
        Compute source representation, knowledge triple representations,
        and score each triple with softmax probability.
        """

        # Encode source (with <s> at beginning)
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        src_enc = encoder_out["encoder_out"][0]  # (T_src, B, C)
        c_s = src_enc[0]  # (B, C) - representation of <s>

        # BUG: from here - about distributed training
        # Encode knowledge
        knw_encoder_out = self.knw_encoder(
            knw_tokens,
            src_lengths=knw_lengths,
            knw_sep=True,
            return_all_hiddens=return_all_hiddens
        )
        knw_enc = knw_encoder_out["encoder_out"][0]  # (T_knw, B, C)
        knw_enc = knw_enc.transpose(0, 1)  # (B, T_knw, C)
        # to here

        # Triple indices: (B, T_knw) → each token's triple id
        triple_indices = knw_encoder_out["triple_indices"][0].transpose(0, 1)  # (B, T_knw)
        # print("Triple indices shape: ", triple_indices.shape)
        triple_indices = triple_indices.to(knw_enc.device)  # ensure same device

        B, T_knw, C = knw_enc.shape
        max_idx = triple_indices.max().item() + 1  # number of triples
        Z = torch.zeros(B, max_idx, C, device=knw_enc.device)  # (B, Z, C)
        count = torch.zeros(B, max_idx, 1, device=knw_enc.device)  # (B, Z, 1)

        # One-hot mask: (B, T_knw, Z)
        triple_one_hot = torch.nn.functional.one_hot(triple_indices, num_classes=max_idx).float()  # (B, T_knw, Z)

        # Mean pooling per triple: matrix multiply to avoid looping
        Z = torch.bmm(triple_one_hot.transpose(1, 2), knw_enc)  # (B, Z, C)
        count = triple_one_hot.sum(dim=1, keepdim=True).transpose(1, 2)  # (B, Z, 1)
        z_mean = Z / (count + 1e-6)  # (B, Z, C)

        # Compute triple scores: dot product between c_s and z_mean
        # c_s: (B, C), z_mean: (B, Z, C) → output: (B, Z)
        scores = torch.bmm(z_mean, c_s.unsqueeze(2)).squeeze(-1)  # (B, Z)
        probs = torch.softmax(scores, dim=-1)  # (B, Z)
        # print("Probs shape: ", probs.shape)

        selected_triple_ids, selected_knw_tokens, selected_knw_lengths = self._sample_triples(
            probs=probs,
            triple_indices=triple_indices,
            knw_tokens=knw_tokens,
            sample_times=sample_times,
            inference=not self.training,
            padding_idx=self.encoder.padding_idx  # or whatever you use
        )

        log_probs = torch.log(probs + 1e-8)  # (B, Z), add epsilon for stability

        # selected_triple_ids: (B, sample_times)
        # Gather log-probs of selected triples
        log_p_t = log_probs.gather(dim=1, index=selected_triple_ids)  # (B, sample_times)

        # Optionally average across multiple samples
        log_p_t = log_p_t.mean(dim=1)  # (B,)


        return {
            "triple_probs": probs,          # p(t_i | X)
            "triple_scores": scores,        # a_i = c_s^T z_i
            "triple_vectors": z_mean,       # z_i for each triple
            "source_repr": c_s,             # c_s
            "encoder_out": encoder_out,
            "knw_encoder_out": knw_encoder_out,
            "selected_triples_ids": selected_triple_ids,
            "selected_knw_tokens": selected_knw_tokens, # (B, T_knw)
            "selected_knw_lengths": selected_knw_lengths, # (B, T_knw) - number of tokens in each triple
            "log_p_t": log_p_t
        }

    def _sample_triples(
        self,
        probs: torch.Tensor,               # (B, Z)
        triple_indices: torch.Tensor,     # (B, T_knw)
        knw_tokens: torch.Tensor,         # (B, T_knw)
        sample_times: int,
        inference: bool = False,
        padding_idx: int = 0              # e.g. <pad> token ID
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples triples and returns token IDs from those triples, padded to original length.

        Returns:
            selected_triples: (B, sample_times)
            selected_knw_tokens: (B, T_knw) — tokens not in sampled triples are replaced with `padding_idx`
        """
        B, Z = probs.shape
        _, T_knw = knw_tokens.shape

        sample_times = min(sample_times, Z)
        # === Step 1: Sample triple indices ===
        if inference:
            selected = torch.topk(probs, k=sample_times, dim=-1).indices  # (B, sample_times)
        else:
            sampled = torch.multinomial(probs, num_samples=sample_times, replacement=True)  # (B, sample_times)
            one_hot = torch.nn.functional.one_hot(sampled, num_classes=Z).sum(dim=1)        # (B, Z)
            selected = torch.topk(one_hot, k=sample_times, dim=-1).indices  # (B, sample_times)

        # === Step 2: Create token-level mask ===
        selected_exp = selected.unsqueeze(-1)             # (B, sample_times, 1)
        triple_indices_exp = triple_indices.unsqueeze(1)  # (B, 1, T_knw)
        match_mask = selected_exp == triple_indices_exp   # (B, sample_times, T_knw)
        token_mask = match_mask.any(dim=1)                # (B, T_knw)

        # === Step 3: Mask knw_tokens (replace non-selected tokens with padding_idx) ===
        selected_knw_tokens = knw_tokens.masked_fill(~token_mask, padding_idx)  # (B, T_knw)
        selected_knw_lengths = token_mask.sum(dim=1)

        return selected, selected_knw_tokens, selected_knw_lengths


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
