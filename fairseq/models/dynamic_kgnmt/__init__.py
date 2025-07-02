# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .dynamic_kgnmt_config import (
    KgNMTConfig,
    KnowledgeSelectorConfig,
    DynamicKgNMTConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)

from .kgnmt_decoder import KgNMTDecoder, KgNMTDecoderBase, Linear
from .kgnmt_encoder import KgNMTEncoder, KgNMTEncoderBase
from .kgnmt_knw_encoder import KgNMTKnowledgeEncoder, KgNMTKnowledgeEncoderBase

from .kgnmt_base import KgNMTModelBase, Embedding
from .knowledge_selector_base import KnowledgeSelectorBase
from .dynamic_kgnmt_base import DynamicKgNMTModelBase

from .kgnmt_legacy import KgNMTModel
from .knowledge_selector_legacy import KnowledgeSelector
from .dynamic_kgnmt_legacy import (
    DynamicKgNMTModel,
    tiny_architecture,
    base_architecture,
    dynamic_kgnmt_iwslt_vi_en,
    dynamic_kgnmt_wmt_en_vi,
)

__all__ = [
    "KgNMTModelBase",
    "KnowledgeSelectorBase",
    "DynamicKgNMTModelBase",
    "KgNMTConfig",
    "KnowledgeSelectorConfig",
    "DynamicKgNMTConfig",
    "KgNMTDecoder",
    "KgNMTDecoderBase",
    "KgNMTEncoder",
    "KgNMTEncoderBase",
    "KgNMTKnowledgeEncoder",
    "KgNMTKnowledgeEncoderBase",
    "KgNMTModel",
    "KnowledgeSelector",
    "DynamicKgNMTModel",
    "Embedding",
    "Linear",
    "base_architecture",
    "tiny_architecture",
    "dynamic_kgnmt_iwslt_vi_en",
    "dynamic_kgnmt_wmt_en_vi",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
]
