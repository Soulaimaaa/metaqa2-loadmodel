from importlib import machinery
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer
from datasets import load_dataset,load_metric
import numpy as np 
import torch
from torch import nn
import sklearn
class MetaQA2_Eval():
    def __init__(self,model_path,eval_data,agents):
        self.model_loaded = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.trainer = Trainer(model = self.model_loaded)
        self.eval_data = load_dataset('json', data_files=eval_data)
        self.qids = self.eval_data["train"]["id"]
        self.agents = agents
        self.predictions = self.get_predictions()

    def preprocess_function(self,examples):
        questions = [q.strip() for q in examples["question_statement_text"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=512,
            truncation="only_second",
            return_offsets_mapping=False,
            padding="max_length",
        )
        return inputs

    def get_predictions(self):
        tokenized_input = self.eval_data.map(self.preprocess_function, batched=True)
        results = self.trainer.predict(test_dataset=tokenized_input['train'])
        predictions, label_ids, metrics = results[0], results[1], results[2]
        # print("label: ",label_ids)
        m = [torch.nn.Softmax(dim=-1)(torch.tensor(p)).tolist() for p in predictions]
        # print("predictions",predictions)
        return m

    def compute_metrics(self,eval_pred,metric):      
        logits = eval_pred
        # print("eval_pred",eval_pred)
        labels = self.eval_data['train']['label']
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    def getConfusionMatrix(self):
        y_true = self.eval_data['train']['label']
        preds = self.predictions
        y_pred = np.argmax(preds, axis=-1)
        print("classfication report:    ", sklearn.metrics.classification_report(y_true, y_pred, labels=[0,1]))
        print("confusion matrix:    ", sklearn.metrics.confusion_matrix(y_true, y_pred))
  

    def get_accuracy(self):
        metric = load_metric("accuracy")
        return self.compute_metrics(self.predictions,metric)

    def get_f1_EM(self):
        eval_data = self.eval_data["train"]
        agents = self.agents
        # only the probability that the hypothesis and the premise are entailed is retrieved
        entail_pred = [p[1] for p in self.predictions]
        
        # map the probabilties of each agent depending on qid
        mappings_agents ={agents[0]:{},agents[1]:{}}
        for i in range(len(self.qids)):
            agent = eval_data['agent'][i]
            qid = self.qids[i]
            gold_ans = eval_data['golden_answer'][i]
            pred = eval_data['answer_text'][i]
            prob_of_entailement = entail_pred[i]
            ########################################
            mappings_agents[agent][qid]={"answer_text":pred,"golden_answer":gold_ans,"entails_pred":prob_of_entailement}
            print("check",mappings_agents)
        
        probAgents_forargmax = []
        ids = []
        for id in self.qids:
            if id not in ids:
                probAgents_forargmax.append([mappings_agents[agents[0]][id]["entails_pred"],mappings_agents[agents[1]][id]["entails_pred"]])
                ids.append(id)
        bestAgent = np.argmax(probAgents_forargmax , axis=-1)    

        ##############################################################################
        # 1) load the metric
        metric = load_metric('squad')
        # 2) get the list of labels in the format of the squad metric
        references = []
        predictions = []
        fortesting = []
        for i in range(len(ids)):
            qid = ids[i]
            ref = {'id': qid , 'answers': {'text': [], 'answer_start': []}} 
            for ans in [mappings_agents[agents[bestAgent[i]]][qid]["golden_answer"]]:
                ref['answers']['text'].append(ans)
                ref['answers']['answer_start'].append(0)
            references.append(ref)
            # 3) get the predictions in the format of the squad metric
            if (mappings_agents[agents[bestAgent[i]]][qid]["entails_pred"]<0.5):
                ans = ""
            else:
                ans = mappings_agents[agents[bestAgent[i]]][qid]["answer_text"]
            predictions.append({'id': qid, 'prediction_text': ans})
            fortesting.append({'id': qid, 'prediction_text': mappings_agents[agents[bestAgent[i]]][qid]["answer_text"]})
        print("before metric: ", list(zip(predictions[:10],fortesting[:10],references[:10])))
        # 4) evaluate the predictions
        results = metric.compute(predictions=predictions, references=references)
        return results



