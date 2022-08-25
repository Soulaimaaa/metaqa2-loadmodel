from __future__ import annotations
import json
from datasets import load_dataset, load_metric
import json
import re
import string

class MetaQA2_UpperBoundEvaluation():
    def __init__(self,dataset,agents):
        self.agents=agents
        self.dataset = dataset
        dataset_path = "/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/evaluation/data/validation/extractive/mrqa/"+dataset+".json"
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
        self.agents_preds = {}
        for agent in agents: 
            preds_path = "/ukp-storage-1/khammari/QA-Verification-Via-NLI/qa_agents/"+agent+"/validation/extractive/"+dataset+"/predict_predictions.json"
            with open(preds_path, 'r') as f:
                self.agents_preds[agent] = json.load(f)
    def normalize_answer(self,s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match_score(self,prediction, ground_truth):
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)

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
        predictions = []
        agents_chosen = []
        trial = dict.fromkeys(self.agents, 0)
        for qid in self.list_qids:
            ref = {'id': qid , 'answers': {'text': [], 'answer_start': []}} 
            golden = ""
            for ans in self.get_list_answer_label(qid):
                golden = ans
                ref['answers']['text'].append(ans)
                ref['answers']['answer_start'].append(0)
            references.append(ref)
        # 3) get the predictions in the format of the squad metric
            
            ans = ""
            expert_agent = "spanbert-large-cased_"+ self.dataset
            if self.exact_match_score(self.agents_preds[expert_agent][qid],golden):
                    ans =self.agents_preds[expert_agent][qid]
                    agents_chosen.append({qid:expert_agent})
                    trial[expert_agent] += 1
            else:
                for agent in self.agents:
                    if self.exact_match_score(self.agents_preds[agent][qid],golden):
                        ans =self.agents_preds[agent][qid]
                        agents_chosen.append({qid:agent})
                        trial[agent] += 1
                        break
            predictions.append({'id': qid, 'prediction_text': ans} )
        # 4) evaluate the predictions
        results = metric.compute(predictions=predictions, references=references)
        for agent in self.agents:
            trial[agent] = round(trial[agent]*100/len(self.list_qids),2)
        return len(self.list_qids),trial,results

    
