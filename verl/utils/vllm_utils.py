# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from msgspec import field
from packaging import version as vs
from vllm.lora.models import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from verl.third_party.vllm import get_version

# --- SUPPORTED MODELS SETUP ---
SUPPORTED_MOE_MODELS = []
try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM
    SUPPORTED_MOE_MODELS.extend([DeepseekV2ForCausalLM, DeepseekV3ForCausalLM])
except ImportError: pass

try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM
    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError: pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM
    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError: pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError: pass

try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration
    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError: pass

def patch_vllm_moe_model_weight_loader(model):
    MLP_ATTR_MAPPING = {MixtralForCausalLM: "block_sparse_moe"}
    DEFAULT_MLP_ATTR = "mlp"
    
    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)): return

    model_attr = getattr(model, "model", None) or getattr(model, "language_model", None)
    if model_attr is None:
        raise ValueError("The provided model does not have a valid 'model' or 'language_model' attribute.")

    for layer in model_attr.layers:
        mlp_attr = MLP_ATTR_MAPPING.get(type(model), DEFAULT_MLP_ATTR)
        mlp = getattr(layer, mlp_attr)
        param_dict = dict(mlp.named_parameters())
        for name, param in param_dict.items():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = mlp.experts.weight_loader

class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)

class VLLMHijack:
    @staticmethod
    def hijack():
        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            try:
                # 1. Setup expected modules (Legacy logic, simplified)
                supported_lora_modules = self._adapter_manager.supported_lora_modules
                packed_modules_mapping = self._adapter_manager.packed_modules_mapping
                expected_lora_modules: List[str] = []
                for module in supported_lora_modules:
                    if module in packed_modules_mapping:
                        expected_lora_modules.extend(packed_modules_mapping[module])
                    else:
                        expected_lora_modules.append(module)
                expected_lora_modules = list(set(expected_lora_modules))

                # 2. Prepare Helper and Paths
                lora_tensors = None
                from vllm.lora.peft_helper import PEFTHelper

                if isinstance(lora_request, TensorLoRARequest):
                    peft_config = lora_request.peft_config
                    lora_tensors = lora_request.lora_tensors
                    peft_helper = PEFTHelper.from_dict(peft_config)
                else:
                    lora_path = get_adapter_absolute_path(lora_request.lora_path)
                    peft_helper = PEFTHelper.from_local_dir(lora_path, self.max_position_embeddings)

                peft_helper.validate_legal(self.lora_config)

                model = self._adapter_manager.model
                hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)

                # --- vLLM 0.12.0 COMPATIBILITY FIX ---
                # REMOVED: embeddings, embedding_modules, embedding_padding_modules, target_embedding_padding
                # REASON: These are now deprecated/removed. vLLM infers them or uses defaults.
                
                if isinstance(lora_request, TensorLoRARequest):
                    lora = self._lora_model_cls.from_lora_tensors(
                        lora_model_id=lora_request.lora_int_id,
                        tensors=lora_tensors,
                        peft_helper=peft_helper,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        weights_mapper=hf_to_vllm_mapper,
                    )
                else:
                    lora = self._lora_model_cls.from_local_checkpoint(
                        lora_path,
                        expected_lora_modules,
                        peft_helper=peft_helper,
                        lora_model_id=lora_request.lora_int_id,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        weights_mapper=hf_to_vllm_mapper,
                    )
                # -----------------------------------------

            except Exception as e:
                raise e

            # Removed the extra_vocab_size check as it causes crashes in 0.12.0
            return lora

        def do_hijack(target_cls, target_method_name, hooking_method):
            setattr(target_cls, target_method_name, hooking_method)

        do_hijack(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)

def is_version_ge(pkg: str = "vllm", minver: str = "0.7.3"):
    return vs.parse(get_version(pkg)) >= vs.parse(minver)
