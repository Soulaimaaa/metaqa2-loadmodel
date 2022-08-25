from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset,load_metric
from tqdm import tqdm


def get_f1_EM(eval_path,agents,agents_chosen):
    # mappings_agents = {}
    # for i in range(len(agents)):
    #     mappings_agents[agents[i]]={}
    # for i in range(len(qids)):
    #     agent = eval_data['agent'][i]
    #     qid = qids[i]
    #     gold_ans = eval_data['golden_answer'][i]
    #     pred = eval_data['answer_text'][i]
    #     ########################################
    #     mappings_agents[agent][qid]={"answer_text":pred,"golden_answer":gold_ans,"label":eval_data['label'][i]}
    # ##############################################################################
    # # 1) load the metric
    # metric = load_metric('squad')
    # # 2) get the list of labels in the format of the squad metric
    # references = []
    # predictions = []
    # norep = []
    # for i in range(int(len(qids))):
    #     if qids[i] not in norep:
    #         qid = qids[i]

    #         # references
    #         ref = {'id': qid , 'answers': {'text': [], 'answer_start': []}} 
    #         ans =  mappings_agents[agents[0]][qid]["golden_answer"]
    #         ref['answers']['text'].append(ans)
    #         ref['answers']['answer_start'].append(0)
    #         references.append(ref)

    #         # predictions
    #         ans = ""
    #         for i in range(len(agents)):
    #             if mappings_agents[agents[i]][qid]["label"]:
    #                 ans =mappings_agents[agents[i]][qid]["answer_text"]
    #                 break
            
    #         predictions.append({'id': qid, 'prediction_text': ans})
    #         norep.append(qid)
    #     else:
    #         continue
    eval_data = load_dataset('json', data_files=eval_path)
    qids = eval_data["train"]["id"]
    eval_data = eval_data["train"]
    mappings_agents = {}
    trial = {}
    for i in range(len(agents)):
        mappings_agents[agents[i]]={}
        trial[agents[i]]=0
    for i in tqdm(range(len(qids))):
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
    norep = []
    for i in tqdm(range(len(qids))):
        if qids[i] not in norep:
            qid = qids[i]
            # references
            ref = {'id': qid , 'answers': {'text': [], 'answer_start': []}} 
            ans =  mappings_agents[agents[0]][qid]["golden_answer"]
            ref['answers']['text'].append(ans)
            ref['answers']['answer_start'].append(0)
            references.append(ref)

            # predictions
            ans = ""
            if mappings_agents["spanbert-large-cased_NaturalQuestionsShort"][qid]["label"]:
                    ans =mappings_agents["spanbert-large-cased_NaturalQuestionsShort"][qid]["answer_text"]
                    agents_chosen.append({qid:"spanbert-large-cased_NaturalQuestionsShort"})
                    trial["spanbert-large-cased_NaturalQuestionsShort"] += 1
            else:
                for j in range(len(agents)):
                    if mappings_agents[agents[j]][qid]["label"]:
                        ans =mappings_agents[agents[j]][qid]["answer_text"]
                        agents_chosen.append({qid:agents[j]})
                        trial[agents[j]] += 1
                        break
            
            predictions.append({'id': qid, 'prediction_text': ans})
            norep.append(qid)
        else:
            continue
    print("before metric: ", list(zip(predictions[:10],references[:10])))
    # 4) evaluate the predictions
    return metric.compute(predictions=predictions, references=references),trial,len(norep)

data_sets =["NaturalQuestionsShort"]
agents = ["spanbert-large-cased_HotpotQA","spanbert-large-cased_NaturalQuestionsShort","spanbert-large-cased_NewsQA","spanbert-large-cased_QAMR","spanbert-large-cased_SearchQA","spanbert-large-cased_SQuAD","spanbert-large-cased_TriviaQA-web","spanbert-large-cased_DuoRC"]                     
agents_chosen =[]

for d in data_sets:
    path = "/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/evaluation/dataForUpperBound/id-question-all-data-for-upper-bound-"+d+".jsonl"
    print("*********************",d,"*********************")
    temp,trial,n = get_f1_EM(path,agents,agents_chosen)
    print(temp)
    with open('/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/NaturalQuestionsShort-agentschosen.txt', 'w') as f:
        for line in agents_chosen:
            f.write(str(line))
            f.write('\n')
        f.write(str(temp))
        f.write(str(n))
        for i in agents:
            f.write(" : ".join([i, str(round(trial[i]*100/n,2))]))
            f.write('\n')
        f.write("percentage of getting no answer: ")
        f.write(str(round((n-len(agents_chosen))/n*100,2)))
