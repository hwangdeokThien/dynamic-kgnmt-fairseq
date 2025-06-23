# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .dynamic_kgnmt_config import (
    KGNMTConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .dynamic_kgnmt_decoder import KGNMTDecoder, KGNMTDecoderBase, Linear
from .dynamic_kgnmt_encoder import KGNMTEncoder, KGNMTEncoderBase
from .dynamic_kgnmt_legacy import (
    KGNMTModel,
    base_architecture,
    tiny_architecture,
    # kgnmt_iwslt_de_en,
    kgnmt_iwslt_vi_en,
    kgnmt_wmt_en_vi,
    # kgnmt_vaswani_wmt_en_de_big,
    # kgnmt_vaswani_wmt_en_fr_big,
    # kgnmt_wmt_en_de_big,
    # kgnmt_wmt_en_de_big_t2t,
)
from .dynamic_kgnmt_base import KGNMTModelBase, Embedding


__all__ = [
    "KGNMTModelBase",
    "KGNMTConfig",
    "KGNMTDecoder",
    "KGNMTDecoderBase",
    "KGNMTEncoder",
    "KGNMTEncoderBase",
    "KGNMTModel",
    "Embedding",
    "Linear",
    "base_architecture",
    "tiny_architecture",
    # "kgnmt_iwslt_de_en",
    "kgnmt_iwslt_vi_en",
    "kgnmt_wmt_en_vi",
    # "kgnmt_vaswani_wmt_en_de_big",
    # "kgnmt_vaswani_wmt_en_fr_big",
    # "kgnmt_wmt_en_de_big",
    # "kgnmt_wmt_en_de_big_t2t",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
]
