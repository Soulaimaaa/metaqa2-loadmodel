import json
import tqdm
predictions_path ="/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/training/nq-nli-eval-TriviaQA+SearchQA-Datasets/nq-nli-eval-triviaQA+SearchQA-TriviaQA.jsonl"
with open(predictions_path) as f:
    data = list(f)
subdata = [json.loads(d) for d in data]
temp = []
lenn = len(subdata)/2
for i in range(int(lenn)):
    q = subdata[i]
    temp.append({"id":q["id"],"question_text":q["question_text"],"answer_text":q["answer_text"],"agent":q["agent"],"golden_answer":q["golden_answer"],"label":q["label"],"context":q["context"],"dataset":q["dataset"],"question_statement_text":q["question_statement_text"]})
    q = subdata[int(len(subdata)/2)+i]
    temp.append({"id":q["id"],"question_text":q["question_text"],"answer_text":q["answer_text"],"agent":q["agent"],"golden_answer":q["golden_answer"],"label":q["label"],"context":q["context"],"dataset":q["dataset"],"question_statement_text":q["question_statement_text"]})

with open("/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/evaluation/correctDev-TriviaQA.jsonl", 'w') as f:
    for i in range(len(temp)):
        f.write(json.dumps(temp[i]) + "\n")