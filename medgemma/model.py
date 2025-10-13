# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from dataclasses import dataclass, field
from swift.llm.template.register import TemplateMeta, register_template
from swift.llm.template.template.gemma import PaliGemmaTemplate
from swift.llm.template.constant import MLLMTemplateType, TemplateType

from swift.llm.model.constant import MLLMModelType
from swift.llm.model.model_arch import ModelArch
from swift.llm.model.register import (Model, 
                                      ModelGroup, 
                                      ModelMeta, 
                                      get_model_tokenizer_multimodal,
                                      register_model)
from swift.llm.model.utils import ModelInfo
from swift.llm.template.utils import Prompt

# MedGemma Template
@dataclass  
class MedGemmaTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(default_factory=lambda: ['user\n{{QUERY}}<eos>\n<bos>assistant'])
    chat_sep: Prompt = field(default_factory=lambda: ['<eos>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<eos>'])
    system_prefix: Prompt = field(default_factory=lambda: ['<bos>system\n{{SYSTEM}}<eos>\n'])
    default_system: Prompt = field(default_factory=lambda: ['You are an expert medical AI assistant.'])

register_template(MedGemmaTemplateMeta(MLLMTemplateType.medgemma_it, template_cls=PaliGemmaTemplate))


def get_model_tokenizer_medgemma_multimodal(model_dir: str,
                                           model_info: ModelInfo,
                                           model_kwargs: Dict[str, Any],
                                           load_model: bool = True,
                                           **kwargs):
    """Get model and tokenizer for MedGemma multimodal models with vision config patching"""
    from transformers import PaliGemmaForConditionalGeneration, AutoConfig
    
    # Monkey patch the config loading to add projection_dim
    original_from_pretrained = AutoConfig.from_pretrained
    
    def patched_from_pretrained(pretrained_model_name_or_path, **kwargs_inner):
        config = original_from_pretrained(pretrained_model_name_or_path, **kwargs_inner)
        
        # Patch vision config for MedGemma
        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
            if not hasattr(vision_config, 'projection_dim') and hasattr(vision_config, 'hidden_size'):
                vision_config.projection_dim = vision_config.hidden_size
        
        return config
    
    # Apply the patch temporarily
    AutoConfig.from_pretrained = patched_from_pretrained
    
    try:
        kwargs['automodel_class'] = kwargs['automodel_class'] or PaliGemmaForConditionalGeneration
        model, processor = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)
    finally:
        # Restore original method
        AutoConfig.from_pretrained = original_from_pretrained
    
    return model, processor


# Register MedGemma models
register_model(
    ModelMeta(
        MLLMModelType.medgemma_it,
        [
            ModelGroup([
                Model('/yinghepool/mabingqi/hf_cache/models--google--medgemma-4b-it', 'google/medgemma-4b-it'),
                # Model('google/medgemma-4b-pt', 'google/medgemma-4b-pt'), 
                # Model('google/medgemma-27b-it', 'google/medgemma-27b-it'),
            ])
        ],
        MLLMTemplateType.medgemma_it,
        get_model_tokenizer_medgemma_multimodal,  # Use specialized MedGemma function
        architectures=['PaliGemmaForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.50'],
        tags=['medical', 'vision', 'multimodal'],
    ))


if __name__ == '__main__':
    from swift.llm import InferRequest, PtEngine, RequestConfig
    
    # Test MedGemma multimodal model
    print("\nTesting MedGemma multimodal model...")
    try:
        infer_request = InferRequest(messages=[{
            'role': 'user',
            'content': 'Describe what you see in this medical image and identify any potential abnormalities.'
        }])
        request_config = RequestConfig(max_tokens=512, temperature=0)
        engine = PtEngine('/yinghepool/mabingqi/hf_cache/models--google--medgemma-4b-it')
        response = engine.infer([infer_request], request_config)
        print(f"Multimodal model response: {response[0].choices[0].message.content}")
    except Exception as e:
        print(f"Multimodal model test failed: {e}")