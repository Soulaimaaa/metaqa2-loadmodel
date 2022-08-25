import json
import logging
import re
import string
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, filename="/ukp-storage-1/khammari/logAgents.txt", filemode='w', format='%(message)s')
agents = ["spanbert-large-cased_SQuAD","spanbert-large-cased_TriviaQA-web"]                     
data_sets =["HotpotQA","NaturalQuestionsShort","NewsQA","QAMR","SearchQA","SQuAD","TriviaQA-web","DuoRC"]
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

def writeData(path):
    id_question_goldenAnswer_mappings = {}
    logging.info('path...')
    with open(path) as f:
        data = json.load(f)
    raw_inputs_squad = data["data"]
    for d in raw_inputs_squad:
        id_question_goldenAnswer_mappings[d['id']]={"question":d["question"],"golden_answer":d["answers"]["text"][0],"context":d["context"]}
    return id_question_goldenAnswer_mappings
mappings_per_dataset = {}
for dataset in data_sets:
    mappings_per_dataset[dataset]=writeData("/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/evaluation/data/validation/extractive/mrqa/"+dataset+".json")


for dataset in tqdm(data_sets):
    temp = []
    for agent in tqdm(agents):
        logging.info('dataset...')
        predictions_path ="/ukp-storage-1/khammari/QA-Verification-Via-NLI/qa_agents/"+ agent +"/validation/extractive/"+dataset+"/predict_predictions.json"
        with open(predictions_path) as f:
            data = json.load(f)
        logging.info('data is loaded...')
        subdata = list(data.keys())
        for i in tqdm(subdata):
            q = mappings_per_dataset[dataset][i]
            temp.append({"id":i,"question_text":q["question"],"answer_text":data[i],"agent":agent,"golden_answer":q["golden_answer"],"label":exact_match_score(q["golden_answer"],data[i]),"context":q["context"],"dataset":dataset})
    
    with open("/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/evaluation/id-question-eval-"+agents[0]+"-"+agents[1]+"-"+dataset+".jsonl", 'w') as f:
        for i in range(len(temp)):
            f.write(json.dumps(temp[i]) + "\n")