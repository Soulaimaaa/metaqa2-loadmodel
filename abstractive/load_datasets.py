from typing import Type
from datasets import load_dataset
from QA_Dataset import NarrativeQA_Dataset,DROP_Dataset,HybridQA_Dataset
from numpy import full
from tqdm import tqdm
import logging
import json
import re
import string
logging.basicConfig(level=logging.INFO, filename="/ukp-storage-1/khammari/logMultipleChoice.txt", filemode='w', format='%(message)s')
############################################################
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    if ((type(prediction) == str) & (type(ground_truth) ==str)):
        return (normalize_answer(prediction) == normalize_answer(ground_truth))
    if ( (type(prediction) == str) & (len(ground_truth)>1)):
        return False
    elif (type(prediction) != str):
        result = True
        for i in prediction:
            result = result & (True in [normalize_answer(i) == normalize_answer(g) for g in ground_truth])
        return result
    return (normalize_answer(prediction) == normalize_answer(ground_truth[0]))

################## Narrative #########################
NarrativeQA = NarrativeQA_Dataset('/ukp-storage-1/khammari/QA-Verification-Via-NLI')._dict_qid2question2answer
temp = []
logging.info('running agent...')
predictions_path ="/ukp-storage-1/khammari/QA-Verification-Via-NLI/qa_agents/bart_adapter_narrativeqa/train/abstractive/NarrativeQA/predictions.json"
with open(predictions_path) as f:
    data = json.load(f)
logging.info('data is loaded...')
subdata = list(data.keys())[:100]
for i in tqdm(subdata):
    q = NarrativeQA[i]
    temp.append({"id":i,"question_text":q["question"],"answer_text":data[i],"agent":"bart_adapter_narrativeqa","golden_answer":q["golden_answer"],"label": True in [exact_match_score(answer,data[i]) for answer in q["golden_answer"]]})
with open("/ukp-storage-1/khammari/QA-Verification-Via-NLI/extractive_data/id-question-"+"bart_adapter_narrativeqa"+".jsonl", 'w') as f:
    for i in range(len(temp)):
        f.write(json.dumps(temp[i]) + "\n")

#################### DROP #############################

DROP = DROP_Dataset('train')._dict_qid2question2answer
temp = []
logging.info('running agent...')
predictions_path ="/ukp-storage-1/khammari/QA-Verification-Via-NLI/qa_agents/TASE/train/abstractive/DROP/predictions.json"
with open(predictions_path) as f:
    data = json.load(f)
logging.info('data is loaded...')
subdata = list(data.keys())[:100]
for i in tqdm(subdata):
    q = DROP[i]
    logging.info(q["golden_answer"])
    logging.info(data[i])
    temp.append({"id":i,"question_text":q["question"],"answer_text":data[i],"agent":"TASE","golden_answer":q["golden_answer"],"label": exact_match_score(data[i],q["golden_answer"])})
with open("/ukp-storage-1/khammari/QA-Verification-Via-NLI/extractive_data/id-question-"+"TASE"+".jsonl", 'w') as f:
    for i in range(len(temp)):
        f.write(json.dumps(temp[i]) + "\n")

##################### Hybrider #########################

temp = []
logging.info('running agent...')
predictions_path ="/ukp-storage-1/khammari/QA-Verification-Via-NLI/qa_agents/hybrider/train/multimodal/HybridQA/predictions.json"
with open(predictions_path) as f:
    data = json.load(f)
logging.info('data is loaded...')
for i in tqdm(data):
    print(i)
    temp.append({"id":i["question_id"],"question_text":i["question"],"answer_text":i["pred"],"agent":"hybrider","golden_answer":i["target"][0],"label": exact_match_score(i["target"][0],i["pred"])})
with open("/ukp-storage-1/khammari/QA-Verification-Via-NLI/extractive_data/id-question-"+"hybrider"+".jsonl", 'w') as f:
    for i in range(len(temp)):
        f.write(json.dumps(temp[i]) + "\n")
