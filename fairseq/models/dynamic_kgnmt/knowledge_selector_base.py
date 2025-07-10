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

    def __init__(self, cfg, encoder, knw_encoder, src_dict, knw_dict, x_token=None):
        super().__init__()
        
        self.encoder = encoder
        self.knw_encoder = knw_encoder
        self.x_token = x_token
        self.src_dict = src_dict
        self.knw_dict = knw_dict
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

        x_token = src_dict.index("<x>")

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

        return cls(cfg, encoder=encoder, knw_encoder=knw_encoder, src_dict=src_dict, knw_dict=knw_dict, x_token=x_token)

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

        # Encode source
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        src_enc = encoder_out["encoder_out"][0]  # (T_src, B, C)
        x_token_mask = src_tokens == self.x_token  # (B, T)
        x_token_idx = x_token_mask.float().argmax(dim=1)  # (B)
        batch_size = src_tokens.size(0)
        batch_indices = torch.arange(batch_size, device=src_tokens.device)
        src_enc_bt = src_enc.transpose(0, 1)
        c_x = src_enc_bt[batch_indices, x_token_idx]  # (B, C)
        del src_enc_bt

        # Encode knowledge
        knw_encoder_out = self.knw_encoder(
            knw_tokens,
            src_lengths=knw_lengths,
            knw_sep=True,
            return_all_hiddens=return_all_hiddens
        )
        knw_enc = knw_encoder_out["encoder_out"][0]  # (T_knw, B, C)
        knw_enc = knw_enc.transpose(0, 1)  # (B, T_knw, C)
        triple_indices = knw_encoder_out["triple_indices"][0].transpose(0, 1)  # (B, T_knw)
        triple_indices = triple_indices.to(knw_enc.device)

        B, T_knw, C = knw_enc.size()
        device = knw_enc.device
        Z = triple_indices.max().item() + 1  # number of triples per example

        # === Efficient triple aggregation using index_add ===
        valid_mask = ~(knw_tokens.eq(self.encoder.padding_idx) | (knw_tokens == self.knw_encoder.knw_sep_idx))  # (B, T_knw)

        flat_enc = knw_enc.reshape(B * T_knw, C)
        flat_indices = triple_indices.reshape(B * T_knw)
        flat_mask = valid_mask.reshape(B * T_knw).float()

        batch_offsets = torch.arange(B, device=device) * Z
        batch_ids = torch.arange(B, device=device).unsqueeze(1).repeat(1, T_knw).reshape(-1)
        global_indices = batch_offsets[batch_ids] + flat_indices  # (B*T_knw,)

        z_sum = torch.zeros(B * Z, C, device=device)
        z_sum.index_add_(0, global_indices, flat_enc)

        z_count = torch.zeros(B * Z, 1, device=device)
        z_count.index_add_(0, global_indices, flat_mask.unsqueeze(1))

        z_mean = z_sum / (z_count + 1e-6)  # safe division
        z_mean = z_mean.reshape(B, Z, C)   # (B, Z, C)

        # === Compute triple scores ===
        scores = torch.bmm(z_mean, c_x.unsqueeze(2)).squeeze(-1)  # (B, Z)
        probs = torch.softmax(scores, dim=-1)                     # (B, Z)

        # === Sample triples ===
        selected_triple_ids, selected_knw_tokens, selected_knw_lengths = self._sample_triples(
            probs=probs,
            triple_indices=triple_indices,
            knw_tokens=knw_tokens,
            sample_times=sample_times,
            inference=not self.training,
            padding_idx=self.encoder.padding_idx
        )

        log_probs = torch.log(probs + 1e-8)  # (B, Z)
        log_p_t = log_probs.gather(dim=1, index=selected_triple_ids)  # (B, sample_times)
        log_p_t = log_p_t.mean(dim=1)  # (B,)
        print("log_p_t requires_grad:", log_p_t.requires_grad)

        return {
            # "triple_probs": probs,          # p(t_i | X)
            # "triple_scores": scores,        # a_i = c_x^T z_i
            # "triple_vectors": z_mean,       # z_i for each triple
            # "source_repr": c_x,             # c_x
            # "encoder_out": encoder_out,
            # "knw_encoder_out": knw_encoder_out,
            # "selected_triples_ids": selected_triple_ids,
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
        padding_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized and memory-efficient sampling of triples.
        """
        B, Z = probs.shape
        T_knw = knw_tokens.size(1)
        sample_times = min(sample_times, Z)

        # Step 1: Sample triple indices
        probs = probs + 1e-8  # avoid zero-rows
        if inference:
            selected = torch.topk(probs, k=sample_times, dim=-1).indices  # (B, sample_times)
        else:
            selected = torch.multinomial(probs, num_samples=sample_times, replacement=True)  # (B, sample_times)

        # Step 2: Create mask to select tokens belonging to sampled triples
        selected_exp = selected.unsqueeze(-1)             # (B, sample_times, 1)
        triple_indices_exp = triple_indices.unsqueeze(1)  # (B, 1, T_knw)
        match_mask = selected_exp == triple_indices_exp   # (B, sample_times, T_knw)
        token_mask = match_mask.any(dim=1)                # (B, T_knw)

        # Step 3: Mask knw_tokens
        selected_knw_tokens_raw = knw_tokens.masked_fill(~token_mask, padding_idx)  # (B, T_knw)

        # Step 4: Get valid lengths
        selected_knw_lengths = selected_knw_tokens_raw.ne(padding_idx).sum(dim=1)
        max_len = selected_knw_lengths.max().item()

        # Truncate to only max valid tokens (like before)
        selected_knw_tokens = self.left_pad_selected_tokens(selected_knw_tokens_raw, padding_idx)
        selected_knw_tokens = selected_knw_tokens[:, -max_len:]

        return selected, selected_knw_tokens, selected_knw_lengths

    def left_pad_selected_tokens(self, token_tensor: torch.Tensor, padding_idx: int):
        """
        Left-pad non-padding tokens in each row of a (B, T) tensor.
        """
        # Step 1: Create a binary mask: 1 for valid token, 0 for padding
        is_valid = token_tensor.ne(padding_idx).int()  # (B, T)

        # Step 2: Get sorting indices that bring valid tokens to the right
        sort_idx = is_valid.argsort(dim=1, descending=False)  # (B, T)

        # Step 3: Gather tokens using those sorted indices
        padded_tokens = torch.gather(token_tensor, dim=1, index=sort_idx)  # (B, T)

        return padded_tokens

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
