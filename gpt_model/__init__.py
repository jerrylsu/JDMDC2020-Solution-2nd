# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "2.5.1"

from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config
from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from .tokenization_bert import BasicTokenizer, BertTokenizer, BertTokenizerFast, WordpieceTokenizer
# Configurations
from .configuration_utils import PretrainedConfig
# Files and general utilities

from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_tf_available,
    is_torch_available,
)

# Modeling
if is_torch_available():
    from .modeling_utils import PreTrainedModel, prune_layer, Conv1D, top_k_top_p_filtering
    from .modeling_gpt2 import (
        GPT2PreTrainedModel,
        GPT2Model,
        GPT2LMHeadModel,
        GPT2DoubleHeadsModel,
        load_tf_weights_in_gpt2,
        GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
    )
    from .modeling_openai import (
        OpenAIGPTPreTrainedModel,
        OpenAIGPTModel,
        OpenAIGPTLMHeadModel,
        OpenAIGPTDoubleHeadsModel,
        load_tf_weights_in_openai_gpt,
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
    )

    # Optimization
    from .optimization import (
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )
