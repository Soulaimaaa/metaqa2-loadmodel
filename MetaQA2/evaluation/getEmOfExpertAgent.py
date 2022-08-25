from __future__ import annotations
import json
import random
from datasets import load_dataset, load_metric
import json
import re
import string
from collections import Counter
import pandas as pd
import os


class MetaQA2_UpperBoundEvaluation():
    def __init__(self,dataset_path,preds_path,agents):
        self.dataset_path = dataset_path
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
        self.list_qids = []
        self._dict_qid2question ={}
        self._dict_qid2list_answer_labels = {}
        for x in self.data['data']:
            self.list_qids.append(x['id'])
            self._dict_qid2question[x['id']] = x['question']
            self._dict_qid2list_answer_labels[x['id']] = x['answers']['text']
        with open(preds_path, 'r') as f:
            self.dict_qid2prediction = json.load(f)
    #dataset_path ="/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/evaluation/data/validation/extractive/mrqa/NewsQA.json"
    #preds_path = "/ukp-storage-1/khammari/QA-Verification-Via-NLI/qa_agents/spanbert-large-cased_TriviaQA-web/validation/extractive/TriviaQA-web/predict_predictions.json"
    
    def get_list_answer_label(self,qid) -> list(str):
            if qid in self._dict_qid2list_answer_labels:
                return self._dict_qid2list_answer_labels[qid]
            else:
                raise ValueError("qid not found in dataset")

    def evaluate(self) -> dict:
        '''
        Input: dict_qid2prediction: dict of qid (str) to prediction (str)
        Output: {'exact_match': value, 'f1': value}
        '''
        # 1) load the metric
        metric = load_metric('squad')
        # 2) get the list of labels in the format of the squad metric
        references = []
        for qid in self.list_qids:
            ref = {'id': qid , 'answers': {'text': [], 'answer_start': []}} 
            for ans in self.get_list_answer_label(qid):
                ref['answers']['text'].append(ans)
                ref['answers']['answer_start'].append(0)
            references.append(ref)
        # 3) get the predictions in the format of the squad metric
        predictions = [{'id': qid, 'prediction_text': pred} for qid, pred in self.dict_qid2prediction.items()]
        # 4) evaluate the predictions
        results = metric.compute(predictions=predictions, references=references)
        return results

    
