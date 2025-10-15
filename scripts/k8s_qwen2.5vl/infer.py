import os
os.environ["VIDEO_MAX_PIXELS"] = "114896"
os.environ["FPS_MAX_FRAMES"] = "70"

import tqdm
import copy
import json
import torch

from swift.llm import PtEngine, RequestConfig, InferRequest


class SwiftInference:
    def __init__(
        self, 
        model_path,
        adapter_path,
        batch_size: int = 8,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        path_prefix=None
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_config = RequestConfig(
            max_tokens=self.max_tokens, temperature=self.temperature
        )
        
        self.engine = PtEngine(
            model_path, 
            adapters=adapter_path,
            max_batch_size=batch_size, attn_impl='flash_attn', 
        )
        
    
    def infer_one_batch(self, batch_data):
        infer_requests = []
        for data in batch_data:
            path_image = data["zst_path"]
            if self.path_prefix is not None:
                path_image = f"{path_image}{self.path_prefix}"
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": path_image,
                        },
                        {"type": "text", "text": "Extract findings and impressions from this medical video."},
                    ],
                }
            ]
            infer_request = InferRequest(
                messages=messages, 
            )
            infer_requests.append(infer_request)
        
        # infer
        resp_list = self.engine.infer(infer_requests, self.request_config, use_tqdm=False)
        results = []
        for data, resp in zip(batch_data, resp_list):
            data = copy.deepcopy(data)
            pred = resp.choices[0].message.content
            data["pred_llm"] = pred
            print("-" * 100)
            print(pred)
            results.append(data)
        return results
    
    
    def infer_all(self, datas):
        results = []
        batch_data_list = [datas[i: i+self.batch_size] for i in range(0, len(datas), self.batch_size)]
        for index, batch_data in tqdm.tqdm(enumerate(batch_data_list), total=len(batch_data_list)):
            print(f"Infer: {index} / {len(batch_data_list)}")
            batch_result = self.infer_one_batch(batch_data)
            results.extend(batch_result)
        return results


    def run_batch_with_path_datas(self, path_datalist, out_dirname):
        with open(path_datalist) as f:
            datalist = [json.loads(s) for s in f.readlines()]
        results = self.infer_all(datalist)
        
        dir_save = os.path.join(self.adapter_path, 'infer_results')
        os.makedirs(dir_save, exist_ok=True)
        with open(os.path.join(dir_save, out_dirname), 'w') as f:
            f.write("\n".join([json.dumps(x, ensure_ascii=False) for x in results]))



if __name__ == "__main__":
    model_path = "/yinghepool/zhangshuheng/models/Qwen2.5-VL-3B-Instruct"
    
    adapter_path = "/yinghepool/mabingqi/ms-swift/output/HeadReport-tiantan_Qwen2.5VL-3B_wwwl_rslora/v0-20251014-084959/checkpoint-4000"
    path_datalist = "/yinghepool/mm-data/report/tiantan/20250926-tiantan10w/tiantan_head_7.9w_meta_stdWindow_clean-test.jsonl"
    out_dirname = "infer-tiantan_head_5k-test.jsonl"
    
    batch_size = 16
    max_token = 2048
    temperature = 0.1
    path_prefix = '-head_wwwl_clip'
    
    inference = SwiftInference(
        model_path=model_path, 
        adapter_path=adapter_path, 
        batch_size=batch_size,
        max_tokens=max_token,
        temperature=temperature,
        path_prefix=path_prefix
    )
    inference.run_batch_with_path_datas(path_datalist, out_dirname)
