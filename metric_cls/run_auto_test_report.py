import copy
import os
from dataclasses import dataclass, field
from typing import List

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from json_repair import repair_json
from loguru import logger
import numpy as np
import pandas as pd
import tqdm

from openai import OpenAI



default_abnorm_list = [
  '囊肿', '水肿', '破入脑室', '硬膜下积液', '硬膜下血肿', '硬膜外血肿', '缺血性白质病变', '脂肪瘤', 
  '脑出血', '脑挫裂伤', '脑梗死', '脑疝', '脑肿物', '脑萎缩', '腔隙性脑梗死', '蛛网膜下腔出血', 
  '软化灶', '钙化', '颅外疝'
]

default_prompt_template = """
根据医生颅脑报告从给定的疾病列表中找到病人所患的疾病，并输出一个疾病组成的json格式list，例如["xxx","xxx"]，不要嵌套list。
注意：
- **重要**请严格输出疾病列表中存在的疾病名称，不要输出不存在的疾病名称。
- 报告中的疾病带解剖部位或位置信息，需去掉解剖部位或位置信息后与疾病列表匹配。
- 在报告内容中疾病名字可能有变化，要简单分析。
- 老年脑等价于脑萎缩。
- 缺血性脑白质病变等价于缺血性白质病变。
- 脑水肿等价于水肿。
- 大脑镰下疝、小脑扁桃体疝、小脑幕切迹疝等价于脑疝。

**疾病列表**：{abnorm_list}
报告: {report}
"""

@dataclass
class ReportPerformanceComputorConfig:
    # llm
    base_url: str = "http://10.0.7.10:12338/v1"
    api_key: str = "EMPTY"
    model: str = "qwen"
    enable_thinking: bool = False
    max_tokens: int = 4096
    temperature: float = 0.0
    
    # abnorm
    abnorm_list: List[str] = field(default_factory=lambda: copy.deepcopy(default_abnorm_list))
    prompt_template: str = field(default_factory=lambda: copy.deepcopy(default_prompt_template))
    
    # name
    gt_name: str = "head_report"
    pred_name: str = "pred_llm"
    case_name: str = "series_instance_uid"
    
    # worker
    max_workers: int = 40


class ReportPerformanceComputor:
    def __init__(self, config: ReportPerformanceComputorConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        
    
    def post_llm(self, prompt: str):
        messages = [
            dict(
                role="user",
                content=prompt
            )
        ]
        res = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            extra_body=dict(
                chat_template_kwargs=dict(
                    enable_thinking=self.config.enable_thinking, # open thinking set to True
                )
            )
        ).choices[0]
        # print(res)
        return res.message


    def extract_aisu_from_report(self, report: str, case: str):
        ret = dict(case=case)
        
        # 判断疾病列表
        prompt = self.config.prompt_template.format(abnorm_list="、".join(self.config.abnorm_list), report=report)
        message = self.post_llm(prompt)
        content = message.content if self.config.enable_thinking else message.reasoning_content
        if content is None:
            content = message.content
        curr_abnorm_list = repair_json(content, return_objects=True)
        for curr_abnorm in curr_abnorm_list:
            if curr_abnorm not in self.config.abnorm_list:
                logger.warning(f"{case}: extracted aisu [{curr_abnorm}] not in abnorm list")
        
        for name in self.config.abnorm_list:
            ret[name] = 1 if name in curr_abnorm_list else 0
        return ret
    
    
    def extract_aisu_batch(self, datas: list):
        infos_gt = []
        infos_pred = []

        if self.config.gt_name != "":
            gt_ps = [(data[self.config.gt_name], data[self.config.case_name]) for data in datas]
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                for info in tqdm.tqdm(executor.map(lambda x: self.extract_aisu_from_report(*x), gt_ps), total=len(gt_ps), desc="gt"):
                    infos_gt.append(info)

        if self.config.pred_name != "":
            pred_ps = [(data[self.config.pred_name], data[self.config.case_name]) for data in datas]
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                for info in tqdm.tqdm(executor.map(lambda x: self.extract_aisu_from_report(*x), pred_ps), total=len(pred_ps), desc="pred"):
                    infos_pred.append(info)
        
        df_gt, df_pred = pd.DataFrame(infos_gt), pd.DataFrame(infos_pred)
                    
        return df_gt, df_pred
        
    
    def compute_performance(self, df_gt: pd.DataFrame, df_pred: pd.DataFrame):
        infos = []
        for name in self.config.abnorm_list:
            curr_gt = np.array(df_gt[name].tolist())
            curr_pred = np.array(df_pred[name].tolist())
            N_tp = ((curr_gt == 1) & (curr_pred == 1)).sum()
            N_tn = ((curr_gt == 0) & (curr_pred == 0)).sum()
            N_fp = ((curr_gt == 0) & (curr_pred == 1)).sum()
            N_fn = ((curr_gt == 1) & (curr_pred == 0)).sum()
            recall = round(N_tp / (N_tp + N_fn + 1e-5), 4)
            precision = round(N_tp / (N_tp + N_fp + 1e-5), 4)
            f1 = round(2 * (precision * recall) / (precision + recall + 1e-6), 4)
            infos.append(dict(
                name=name,
                gt=N_tp + N_fn,
                tp=N_tp,
                fp=N_fp,
                fn=N_fn,
                tn=N_tn,
                recall=recall,
                precision=precision,
                f1=f1
            ))
        
        # 先创建DataFrame用于计算macro mean
        df_out = pd.DataFrame(infos)
        
        # Macro mean: 所有类别指标的平均值
        macro_recall = round(df_out["recall"].mean(), 4)
        macro_precision = round(df_out["precision"].mean(), 4)
        macro_f1 = round(df_out["f1"].mean(), 4)
        
        # Micro mean: 所有类别的TP、FP、FN、TN加总后计算
        total_tp = df_out["tp"].sum()
        total_fp = df_out["fp"].sum()
        total_fn = df_out["fn"].sum()
        total_tn = df_out["tn"].sum()
        total_recall = round(total_tp / (total_tp + total_fn + 1e-5), 4)
        total_precision = round(total_tp / (total_tp + total_fp + 1e-5), 4)
        total_f1 = round(2 * (total_precision * total_recall) / (total_precision + total_recall + 1e-6), 4)
        
        # 添加汇总行
        infos.append(dict(
            name="micro_mean",
            gt=total_tp + total_fn,
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            tn=total_tn,
            recall=total_recall,
            precision=total_precision,
            f1=total_f1
        ))
        
        infos.append(dict(
            name="macro_mean",
            gt="-",
            tp="-",
            fp="-",
            fn="-",
            tn="-",
            recall=macro_recall,
            precision=macro_precision,
            f1=macro_f1
        ))
        
        # 重新创建包含所有行的DataFrame
        df_out = pd.DataFrame(infos)
        return df_out
    
    def deal_one_with_path_jsonl(self, path_jsonl: str, is_save: bool=True):
        assert path_jsonl.endswith(".jsonl")
        with open(path_jsonl) as f:
            datas = [json.loads(x) for x in f.readlines()]
            
        df_gt, df_pred = self.extract_aisu_batch(datas)
        df_performance = self.compute_performance(df_gt, df_pred)
        
        if is_save is True:
            df_gt.to_csv(path_jsonl.replace('.jsonl', f'-aisu-{self.config.gt_name}.csv'), index=None)
            df_pred.to_csv(path_jsonl.replace('.jsonl', f'-aisu-{self.config.pred_name}.csv'), index=None)
        
        return df_performance, df_gt, df_pred
