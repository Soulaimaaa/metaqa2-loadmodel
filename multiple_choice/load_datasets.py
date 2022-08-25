from datasets import load_dataset
from QA_Dataset import MultipleChoice_QA_Dataset,RACE_Dataset, CommonSenseQA_Dataset, HellaSWAG_Dataset, SIQA_Dataset, BoolQ_Dataset, List_QA_Datasets
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
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

################## Multiple Choice #########################
boolQA = BoolQ_Dataset('train')._dict_qid2question2answer
CommonSenseQA = CommonSenseQA_Dataset('train')._dict_qid2question2answer
SIQA = SIQA_Dataset('train')._dict_qid2question2answer
HellaSWAG = HellaSWAG_Dataset('train')._dict_qid2question2answer
RACE = RACE_Dataset('all','train')._dict_qid2question2answer
############################################################
mappings_per_dataset= {"BoolQ":boolQA,"CommonSenseQA":CommonSenseQA,"SIQA":SIQA,"HellaSWAG":HellaSWAG,"RACE":RACE}
############################################################
data_sets = ["BoolQ","CommonSenseQA","SIQA","HellaSWAG","RACE"]
agents = ["lewtun_bert-large-uncased-wwm-finetuned-boolq","roberta-large_SIQA","LIAMF-USP_roberta-large-finetuned-race","danlou-albert-xxlarge-v2-finetuned-csqa","prajjwal1-roberta_hellaswag"]
for agent in tqdm(agents):
    temp = []
    logging.info('running agent...')
    if agent != "lewtun_bert-large-uncased-wwm-finetuned-boolq":
        for dataset in tqdm(data_sets):
            logging.info('dataset...')
            predictions_path ="/ukp-storage-1/khammari/QA-Verification-Via-NLI/qa_agents/"+ agent +"/train/multiple_choice/"+dataset+"/predict_predictions.json"
            with open(predictions_path) as f:
                data = json.load(f)
            logging.info('data is loaded...')
            subdata = list(data.keys())[:100]
            for i in tqdm(subdata): 
                q = mappings_per_dataset[dataset][i]
                temp.append({"id":i,"question_text":q["question"],"answer_text":data[i]['text'],"agent":agent,"golden_answer":q["golden_answer"],"label":exact_match_score(q["golden_answer"],data[i]['text'])})
        with open("/ukp-storage-1/khammari/QA-Verification-Via-NLI/extractive_data/id-question-"+agent+".jsonl", 'w') as f:
            for i in range(len(temp)):
                f.write(json.dumps(temp[i]) + "\n")

    else:
        predictions_path = "/ukp-storage-1/khammari/QA-Verification-Via-NLI/qa_agents/lewtun_bert-large-uncased-wwm-finetuned-boolq/train/multiple_choice/BoolQ/seq_clas_predict_predictions.json"
        with open(predictions_path) as f:
            data = json.load(f)
        subdata = list(data.keys())[:100]
        for i in tqdm(subdata):
            q = mappings_per_dataset["BoolQ"][i]
            temp.append({"id":i,"question_text":q["question"],"answer_text":data[i]['pred'],"agent":agent,"golden_answer":q["golden_answer"],"label":exact_match_score(q["golden_answer"],data[i]['pred'])})
        with open("/ukp-storage-1/khammari/QA-Verification-Via-NLI/extractive_data/id-question-"+agent+".jsonl", 'w') as f:
            for i in range(len(temp)):
                f.write(json.dumps(temp[i]) + "\n")

