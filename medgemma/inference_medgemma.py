#!/usr/bin/env python3
"""
MedGemma推理示例
使用微调后的MedGemma模型进行医疗问答
"""

from swift.llm import InferRequest, PtEngine, RequestConfig
from PIL import Image
import argparse


def main():
    parser = argparse.ArgumentParser(description="MedGemma Inference Example")
    parser.add_argument("--model-path", default="google/medgemma-4b-it", 
                       help="Path to MedGemma model")
    parser.add_argument("--question", default="What are the symptoms of pneumonia?",
                       help="Medical question to ask")
    parser.add_argument("--image-path", default=None,
                       help="Path to medical image (optional)")
    args = parser.parse_args()
    
    print(f"Loading MedGemma model: {args.model_path}")
    
    # 初始化MedGemma模型
    engine = PtEngine(args.model_path)
    
    # 构建推理请求
    messages = [
        {
            "role": "system",
            "content": "You are an expert medical AI assistant. Provide accurate, helpful medical information."
        },
        {
            "role": "user", 
            "content": args.question
        }
    ]
    
    # 如果提供了图像，添加到请求中
    images = None
    if args.image_path:
        try:
            image = Image.open(args.image_path)
            images = [image]
            print(f"Loaded medical image: {args.image_path}")
        except Exception as e:
            print(f"Failed to load image: {e}")
            return
    
    # 创建推理请求
    infer_request = InferRequest(
        messages=messages,
        images=images
    )
    
    # 配置推理参数
    request_config = RequestConfig(
        max_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True
    )
    
    print(f"\nQuestion: {args.question}")
    if images:
        print("Image: Provided")
    
    # 执行推理
    print("\nGenerating response...")
    try:
        response = engine.infer([infer_request], request_config)
        answer = response[0].choices[0].message.content
        
        print(f"\nMedGemma Response:")
        print("=" * 60)
        print(answer)
        print("=" * 60)
        
    except Exception as e:
        print(f"Inference failed: {e}")


if __name__ == "__main__":
    main()
