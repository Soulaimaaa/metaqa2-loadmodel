from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer
from datasets import load_dataset
from datasets import load_metric
import numpy as np

datasetss = ["DuoRC","HotpotQA","NaturalQuestionsShort","NewsQA","QAMR","SearchQA","SQuAD","TriviaQA-web"]
model_loaded = AutoModelForSequenceClassification.from_pretrained("/ukp-storage-1/khammari/BertModelNLI_SearchQA_Trivia_4K")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
trainer = Trainer(model = model_loaded)
predictions_path = "/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/evaluation/nq-nli-eval-triviaQA+SearchQA-SearchQA.jsonl"
dataset = load_dataset('json', data_files=predictions_path)
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question_statement_text"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=False,
        padding="max_length",
    )
    return inputs

tokenized_input = dataset.map(preprocess_function, batched=True)
results = trainer.predict(test_dataset=tokenized_input['train'])
predictions, label_ids, metrics = results[0], results[1], results[2]

##############################################################################
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits = eval_pred
    labels = dataset['train']['label']
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)
print("accuracy:    ",compute_metrics(predictions))

##############################################################################
entail_pred = [p[1] for p in predictions]
l =dataset["train"]
tmp ={"spanbert-large-cased_SearchQA":{},"spanbert-large-cased_TriviaQA-web":{}}
for i in range(len(dataset["train"]["id"])):
    tmp[l['agent'][i]][l["id"][i]]={"answer_text":l['answer_text'][i],"golden_answer":l['golden_answer'][i],"entails_pred":entail_pred[i]}
forargmamx = []
agents = ["spanbert-large-cased_SearchQA","spanbert-large-cased_TriviaQA-web"]
for id in dataset["train"]["id"]:
    forargmamx.append([tmp[agents[0]][id]["entails_pred"],tmp[agents[1]][id]["entails_pred"]])
bestAgent = np.argmax(forargmamx, axis=-1)    
##############################################################################
# 1) load the metric
metric = load_metric('squad')
# 2) get the list of labels in the format of the squad metric
references = []
predictions = []
for i in range(len(dataset["train"]["id"])):
    qid = dataset["train"]["id"][i]
    ref = {'id': qid , 'answers': {'text': [], 'answer_start': []}} 
    for ans in tmp[agents[bestAgent[i]]][qid]["golden_answer"]:
        ref['answers']['text'].append(ans)
        ref['answers']['answer_start'].append(0)
    references.append(ref)
    # 3) get the predictions in the format of the squad metric
    if (tmp[agents[bestAgent[i]]][qid]["entails_pred"]<0.5):
        ans = ""
    else:
        ans = tmp[agents[bestAgent[i]]][qid]["answer_text"]
    predictions.append({'id': qid, 'prediction_text': ans})
# 4) evaluate the predictions
results = metric.compute(predictions=predictions, references=references)
##############################################################################
print(results)
