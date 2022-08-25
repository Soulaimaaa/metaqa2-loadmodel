import json
import logging
from tqdm import tqdm
agent = "spanbert-large-cased_HotpotQA"
data_set = "SearchQA"
logging.basicConfig(level=logging.INFO, filename="/ukp-storage-1/khammari/log-"+ agent +"-" + data_set+ ".txt", filemode='w', format='%(message)s')

dictOfResults = {}
results = []
def writeData(path,j):
    logging.info('path...')
    with open(path) as f:
        data = json.load(f)
    raw_inputs_squad = data["data"]
    for i,d in enumerate(raw_inputs_squad):
        results.append({"id":d['id'],"question":d["question"],"golden_answer":d["answers"]["text"][0]})
    return results

writeData("/ukp-storage-1/khammari/QA-Verification-Via-NLI/extractive_data/mrqa/"+data_set+".json",len(results))

temp = []
logging.info('running agent...')
logging.info('dataset...')
predictions_path ="/ukp-storage-1/khammari/QA-Verification-Via-NLI/qa_agents/"+ agent +"/train/extractive/"+data_set+"/predict_predictions.json"
with open(predictions_path) as f:
    data = json.load(f)
logging.info('data is loaded...')
for i in tqdm(data.keys()):
    q = [x for x in results if x['id']==i][0]
    temp.append({"id":i,"question_text":q["question"],"answer_text":data[i],"agent":agent,"golden_answer":q["golden_answer"],"label":q["golden_answer"]==data[i]})
with open("/ukp-storage-1/khammari/QA-Verification-Via-NLI/extractive_data/id-question-"+agent+"-"+data_set+".jsonl", 'w') as f:
    for i in tqdm(range(len(temp))):
        f.write(json.dumps(temp[i]) + "\n")


