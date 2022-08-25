import json
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, filename="/ukp-storage-1/khammari/logAgents.txt", filemode='w', format='%(message)s')
agents = ["spanbert-large-cased_HotpotQA","spanbert-large-cased_NaturalQuestionsShort","spanbert-large-cased_NewsQA","spanbert-large-cased_QAMR","spanbert-large-cased_SearchQA","spanbert-large-cased_SQuAD","spanbert-large-cased_TriviaQA-web","spanbert-large-cased_DuoRC"]                     
data_sets =["HotpotQA","NaturalQuestionsShort","NewsQA","QAMR","SearchQA","SQuAD","TriviaQA-web","DuoRC"]

def writeData(path):
    id_question_goldenAnswer_mappings = []
    logging.info('path...')
    with open(path) as f:
        data = json.load(f)
    raw_inputs_squad = data["data"]
    for d in raw_inputs_squad:
        id_question_goldenAnswer_mappings.append({"id":d['id'],"question_text":d["question"],"answer_text":d["answers"]["text"][0],"context":d["context"]})
    return id_question_goldenAnswer_mappings
for dataset in data_sets:
    temp=writeData("/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/DataPreprocessing/loadData/mrqa/"+dataset+".json")
    with open("/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/evaluation/DataForEvaluation/id-question-correct-"+dataset+".jsonl", 'w') as f:
            for i in range(len(temp)):
                f.write(json.dumps(temp[i]) + "\n")