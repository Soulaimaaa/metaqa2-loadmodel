from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset,load_metric



def get_f1_EM(dataset):
    eval_path ="/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/training/nq-nli-eval-TriviaQA+SearchQA-Datasets/nq-nli-eval-triviaQA+SearchQA-"+dataset+".jsonl"
    eval_data = load_dataset('json', data_files=eval_path)
    qids = eval_data["train"]["id"]
    agents = ["spanbert-large-cased_SearchQA","spanbert-large-cased_TriviaQA-web"]
    eval_data = eval_data["train"]
    mappings_agents ={agents[0]:{},agents[1]:{}}
    for i in range(len(qids)):
        agent = eval_data['agent'][i]
        qid = qids[i]
        gold_ans = eval_data['golden_answer'][i]
        pred = eval_data['answer_text'][i]
        ########################################
        mappings_agents[agent][qid]={"answer_text":pred,"golden_answer":gold_ans,"label":eval_data['label'][i]}
    ##############################################################################
    # 1) load the metric
    metric = load_metric('squad')
    # 2) get the list of labels in the format of the squad metric
    references = []
    predictions = []
    for i in range(int(len(qids)/2)):
        qid = qids[i]

        # references
        ref = {'id': qid , 'answers': {'text': [], 'answer_start': []}} 
        ans =  mappings_agents[agents[0]][qid]["golden_answer"]
        ref['answers']['text'].append(ans)
        ref['answers']['answer_start'].append(0)
        references.append(ref)

        # predictions
        if (not mappings_agents[agents[0]][qid]["label"] and not mappings_agents[agents[1]][qid]["label"]):
            ans = ""
        elif (mappings_agents[agents[0]][qid]["label"]):
            ans =mappings_agents[agents[0]][qid]["answer_text"]
        elif (mappings_agents[agents[1]][qid]["label"]):
            ans =mappings_agents[agents[1]][qid]["answer_text"]
        
        predictions.append({'id': qid, 'prediction_text': ans})

    print("before metric: ", list(zip(predictions[:10],references[:10])))
    print("metrics:",metric.compute(predictions=predictions[:10], references=references[:10]))
    # 4) evaluate the predictions
    for i in range(len(predictions)):
        m =  metric.compute(predictions=[predictions[i]], references=[references[i]])
        if  m["exact_match"] != m["f1"]:
            print(predictions[i], references[i],m, "\n")

data_sets = ["NewsQA"]
for d in data_sets:
    print("*********************",d,"*********************")
    get_f1_EM(d)