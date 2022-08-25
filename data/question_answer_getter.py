import json

results = {}
def writeData(path,j):
    with open(path) as f:
        data = json.load(f)
    raw_inputs_squad = data["data"]
    for i,d in enumerate(raw_inputs_squad):
        results[i+j]={"question_text":d["question"],"answer_text":d["answers"]["text"][0]}
    return results
j = 0
squadData = writeData("/ukp-storage-1/khammari/QA-Verification-Via-NLI/data/nq-nli-SQUAD.json",j)
triviaData = writeData("/ukp-storage-1/khammari/QA-Verification-Via-NLI/data/nq-nli-TRIVIA.json",j+len(results)) 
searchQAData = writeData("/ukp-storage-1/khammari/QA-Verification-Via-NLI/data/nq-nli-searchQA.json",j+len(results))
newsQAData = writeData("/ukp-storage-1/khammari/QA-Verification-Via-NLI/data/nq-nli-newsQA.json",j+len(results))
naturalQuestionShortData = writeData("/ukp-storage-1/khammari/QA-Verification-Via-NLI/data/nq-nli-naturalQuestionShort.json",j+len(results))
hotpotData = writeData("/ukp-storage-1/khammari/QA-Verification-Via-NLI/data/nq-nli-hotpotQA.json",j+len(results))
duoRCData = writeData("/ukp-storage-1/khammari/QA-Verification-Via-NLI/data/nq-nli-duoRC.json",j+len(results))
QAMRData = writeData("/ukp-storage-1/khammari/QA-Verification-Via-NLI/data/nq-nli-QAMR.json",j+len(results))
with open("/ukp-storage-1/khammari/QA-Verification-Via-NLI/data/processed-input.jsonl", 'w') as f:
    for i in range(len(results)):
        f.write(json.dumps(results[i]) + "\n")