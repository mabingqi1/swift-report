# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional
from swift.llm.dataset.register import DatasetMeta, SubsetDataset, register_dataset
from swift.llm.dataset.preprocessor import MessagesPreprocessor
import json
import os


class MedGemmaPreprocessor(MessagesPreprocessor):
    """
    MedGemma数据预处理器，支持SLAKE医疗视觉问答数据集
    适用于医疗多模态对话数据，支持图像+文本的输入格式
    """
    
    def __init__(self, **kwargs):
        # MedGemma使用标准的消息格式
        super().__init__(
            role_key='role',
            content_key='content', 
            user_role='user',
            assistant_role='assistant',
            system_role='system',
            **kwargs
        )
    
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        预处理单个样本，支持SLAKE数据集和其他MedGemma多模态输入格式
        
        支持的输入格式:
        - SLAKE格式 (question, answer, img_name, img_organ等字段)
        - 医疗问答格式 (question/answer字段)
        - messages格式 (已处理的对话格式)
        """
        
        # 如果已经是messages格式，直接处理其中的媒体类型
        if 'messages' in row:
            messages = row['messages']
            # 处理消息中的特殊媒体类型
            for message in messages:
                if isinstance(message.get('content'), list):
                    for content_item in message['content']:
                        if isinstance(content_item, dict) and content_item.get('type'):
                            # 将医疗特殊图像类型统一处理为image类型
                            if content_item['type'] in ['image3d', 'ct', 'mri', 'xray', 'dicom']:
                                content_item['type'] = 'image'
                                # 如果有对应的字段，重命名
                                if content_item.get('image3d'):
                                    content_item['image'] = content_item.pop('image3d')
                                elif content_item.get('ct'):
                                    content_item['image'] = content_item.pop('ct')
                                elif content_item.get('mri'):
                                    content_item['image'] = content_item.pop('mri')
                                elif content_item.get('xray'):
                                    content_item['image'] = content_item.pop('xray')
                                elif content_item.get('dicom'):
                                    content_item['image'] = content_item.pop('dicom')
            return row
        
        # 处理SLAKE数据集格式
        if 'question' in row and ('answer' in row or 'label' in row):
            question = row['question']
            # SLAKE数据集中answer和label字段都可能包含答案
            answer = row.get('answer') or row.get('label')
            
            if question is None or answer is None:
                return None
            
            # 构建用户消息内容
            user_content = []
            
            # 添加文本内容
            user_content.append({"type": "text", "text": str(question)})
            
            # 处理图像内容 - SLAKE数据集支持多种图像字段
            image_data = None
            
            # 优先使用base64编码的图像
            if row.get('base64'):
                image_data = row['base64']
            elif row.get('img_name'):
                # 如果有图像文件名，使用文件名
                image_data = row['img_name']
            elif row.get('image'):
                image_data = row['image']
            elif row.get('img'):
                image_data = row['img']
            
            # 添加图像到用户内容
            if image_data:
                user_content.append({"type": "image", "image": image_data})
            
            # 构建系统提示词，根据数据集特点进行优化
            img_organ = row.get('img_organ', '')
            answer_type = row.get('answer_type', '')
            q_lang = row.get('q_lang', 'en')
            
            # 构建系统消息
            system_content = "You are an expert medical AI assistant specialized in medical visual question answering."
            if img_organ:
                system_content += f" You are analyzing medical images of {img_organ.lower()}."
            if answer_type == 'CLOSED':
                system_content += " Please provide concise yes/no or short factual answers."
            elif answer_type == 'OPEN':
                system_content += " Please provide detailed and comprehensive answers."
            system_content += " Be accurate and professional in your medical assessments."
            
            # 构建消息格式
            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user", 
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": str(answer)
                }
            ]
            
            # 保留重要的元数据
            result = {}
            # 保留SLAKE数据集的重要字段
            if 'qid' in row:
                result['qid'] = row['qid']
            if 'img_organ' in row:
                result['img_organ'] = row['img_organ']
            if 'answer_type' in row:
                result['answer_type'] = row['answer_type']
            if 'q_lang' in row:
                result['q_lang'] = row['q_lang']
            if 'content_type' in row:
                result['content_type'] = row['content_type']
            
            result['messages'] = messages
            return result
        
        # 处理通用question/answer格式 (常见的医疗问答格式)
        question = row.get('question') or row.get('query') or row.get('input')
        answer = row.get('answer') or row.get('response') or row.get('output')
        
        if question is not None and answer is not None:
            # 处理图像内容 - 支持多种医疗图像类型
            images = (row.get('image') or row.get('image3d') or row.get('ct') or 
                     row.get('mri') or row.get('xray') or row.get('dicom'))
            if images and not isinstance(images, list):
                images = [images]
            
            # 构建用户消息内容
            user_content = []
            
            # 添加文本内容
            if isinstance(question, str):
                user_content.append({"type": "text", "text": question})
            
            # 添加图像内容 (如果有的话)
            if images:
                for img in images:
                    if isinstance(img, dict):
                        user_content.append({"type": "image", "image": img})
                    else:
                        # 假设是图像路径或bytes
                        user_content.append({"type": "image", "image": img})
            
            # 构建消息格式
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert medical AI assistant. Provide accurate, helpful medical information while being clear about limitations."
                },
                {
                    "role": "user", 
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
            
            # 保留其他字段
            excluded_fields = ['question', 'query', 'input', 'answer', 'response', 'output', 
                             'image', 'image3d', 'ct', 'mri', 'xray', 'dicom']
            result = {k: v for k, v in row.items() if k not in excluded_fields}
            result['messages'] = messages
            
            return result
        
        # 如果格式不匹配，返回None跳过该样本
        return None


# 注册MedGemma数据集
def register_yh_datasets():
    """注册MedGemma相关的数据集"""
    register_dataset(
        DatasetMeta(
            dataset_name='medgemma-yh-train',
            ms_dataset_id=None,
            hf_dataset_id=None,
            dataset_path=None,
            subsets=['default'],
            split=['train'],
            preprocess_func=MedGemmaPreprocessor(),
            tags=['medical', 'vision', 'multimodal', 'vqa'],
            help='Medical visual question-answering dataset for MedGemma fine-tuning'
        ), exist_ok=True
    )
    register_dataset(
        DatasetMeta(
            dataset_name='medgemma-yh-val',
            ms_dataset_id=None,
            hf_dataset_id=None,
            dataset_path=None,
            subsets=['default'],
            split=['test'],
            preprocess_func=MedGemmaPreprocessor(),
            tags=['medical', 'vision', 'multimodal', 'vqa'],
            help='Medical visual question-answering dataset for MedGemma fine-tuning'
        ), exist_ok=True
    )

def register_slake_datasets():
    """注册SLAKE医疗视觉问答数据集"""
    register_dataset(
        DatasetMeta(
            dataset_name='slake-train',
            hf_dataset_id='BoKelvin/SLAKE',
            preprocess_func=MedGemmaPreprocessor(),
            tags=['medical', 'vision', 'multimodal', 'vqa', 'slake'],
            split=['train'],
        ), exist_ok=True)
    
    register_dataset(
        DatasetMeta(
            dataset_name='slake-val',
            hf_dataset_id='BoKelvin/SLAKE',
            preprocess_func=MedGemmaPreprocessor(),
            tags=['medical', 'vision', 'multimodal', 'vqa', 'slake'],
            split=['validation'],
        ), exist_ok=True)
    
    register_dataset(
        DatasetMeta(
            dataset_name='slake-test',
            hf_dataset_id='BoKelvin/SLAKE',
            preprocess_func=MedGemmaPreprocessor(),
            tags=['medical', 'vision', 'multimodal', 'vqa', 'slake'],
            split=['test'],
        ), exist_ok=True)

register_yh_datasets()
register_slake_datasets()
print("[MedGemma] YH dataset and preprocessor registered successfully!")
print("[MedGemma] SLAKE medical VQA dataset registered successfully!")