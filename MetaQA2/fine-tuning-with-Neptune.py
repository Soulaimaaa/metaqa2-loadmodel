from transformers import get_scheduler,DefaultDataCollator, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AdamW
from datasets import load_dataset,load_metric
import neptune.new as neptune
from neptune.new.integrations.python_logger import NeptuneHandler
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml
import logging
import numpy as np
import torch
from torchsummary import summary 

path_for_config='/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/config.yaml'
with open(path_for_config, 'r') as file:
    config = yaml.safe_load(file)
predictions_path = config['paths']['train_data_path']
eval_path = config['paths']['dev_data_path']


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    filename="/ukp-storage-1/khammari/log-train_model_with_LR_13_08_2.txt")
logger = logging.getLogger(__name__)

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
    inputs["label"] = [1 if label else 0 for label in examples['label']]
    return inputs

def compute_metrics(preds,labels):
    metric = load_metric("accuracy")
    return metric.compute(predictions=preds, references=labels)

def get_f1_EM(eval_data,agents,predictions):
        qids = eval_data["id"]
        # only the probability that the hypothesis and the premise are entailed is retrieved
        probs_after_softmax = [torch.nn.Softmax(dim=-1)(torch.tensor(p)).tolist() for p in predictions]
        entail_pred = [p[1] for p in probs_after_softmax]
        # map the probabilties of each agent depending on qid
        mappings_agents ={agents[0]:{},agents[1]:{}}

        for i in range(len(qids)):
            agent = eval_data['agent'][i]
            qid = qids[i]
            gold_ans = eval_data['golden_answer'][i]
            pred = eval_data['answer_text'][i]
            prob_of_entailement = entail_pred[i]
            ########################################
            mappings_agents[agent][qid]={"answer_text":pred,"golden_answer":gold_ans,"entails_pred":prob_of_entailement}
        
        probAgents_forargmax = []
        ids = []
        for id in qids:
            if id not in ids:
                probAgents_forargmax.append([mappings_agents[agents[0]][id]["entails_pred"],mappings_agents[agents[1]][id]["entails_pred"]])
                ids.append(id)
        bestAgent = torch.argmax(torch.Tensor(probAgents_forargmax) , dim=1)    

        ##############################################################################
        # 1) load the metric
        metric = load_metric('squad')
        # 2) get the list of labels in the format of the squad metric
        references = []
        predictions = []
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
        # 4) evaluate the predictions
        results = metric.compute(predictions=predictions, references=references)
        return results["exact_match"], results["f1"]

def evaluate(preds,labels,b,agents,logits):
    acc= compute_metrics(preds,labels)["accuracy"]
    em,f1=  get_f1_EM(b,agents,logits)
    preds =  np.argmax([l.tolist() for l in logits], axis=-1)
    return acc, em, f1, preds

def mycollator(batch):
    result = {}
    for k in batch[0].keys():
        if k == "labels":
            result[k] = torch.tensor([1 if x[k] else 0 for x in batch])
        elif k in ["answer_text","golden_answer","id","agent"]:
            result[k] = [x[k] for x in batch]
        else:
            result[k] = torch.tensor([x[k] for x in batch])
    return result

if __name__ == "__main__":    
    #  Must be a list of str. Tags of the run. They are editable after run is created. 
    # Tags are displayed in the run's Details and can be viewed in Runs table view as a column.
    run = neptune.init(tags=["SQuAD+TriviaQA"], project="soulaima/fine-tuning") # https://docs.neptune.ai/getting-started/hello-world
    
    logger.addHandler(NeptuneHandler(run=run))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # todo: change hardcoding params and use a yaml file for config (see metaqa repo)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)
    dataset = load_dataset('json', data_files=predictions_path)
    evalDataset = load_dataset('json', data_files=eval_path)
    agents = ["spanbert-large-cased_SearchQA","spanbert-large-cased_TriviaQA-web"]

    tokenized_input = dataset.map(preprocess_function, batched=True, remove_columns=['id', 'question_text', 'answer_text', 'agent','golden_answer','context', 'dataset', 'question_statement_text'])
    tokenized_eval_input = evalDataset.map(preprocess_function, batched=True, remove_columns=['question_text', 'context', 'dataset', 'question_statement_text'])
    tokenized_input = tokenized_input.rename_column("label", "labels")
    tokenized_eval_input = tokenized_eval_input.rename_column("label", "labels")

    #tokenized_input.with_format("torch")
    #tokenized_eval_input.with_format("torch", columns=["labels"], output_all_columns=True) 
    #tokenized_eval_input.set_format("torch")
    #data_collator = DefaultDataCollator()

    train_dataloader = DataLoader(tokenized_input["train"], shuffle=True, batch_size=8, collate_fn=mycollator)
    eval_dataloader = DataLoader(tokenized_eval_input["train"], batch_size=8, collate_fn=mycollator)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
    )

    optimizer = AdamW(model.parameters(), lr=1e-5)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=config['model_config']['num_epochs'] * len(train_dataloader))
    progress_bar = tqdm(range(config['model_config']['num_epochs'] * len(train_dataloader)))

    # change Trainer to for-loop to do Acc, EM, F1 evaluation every X training steps
    logging.info('Training...')
    
     # model.train() sets the mode to train
    for epoch in range(config['model_config']['num_epochs']): # should be 1
        model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"],labels= batch['labels'],attention_mask=batch["attention_mask"])
            
            logits_try = outputs.logits
            predictions_try = torch.argmax(logits_try, dim=-1)
            i = 0
            test_preds = predictions_try.tolist()
            test_labels = batch['labels'].tolist()
            print(test_preds)
            print(test_labels)
            for j in range(len(test_labels)):
                if test_labels[j] != test_preds[j]:
                    i +=1
            print("see loss ",i/len(test_labels))
            # vgg = model.vgg16()
            # print(summary(vgg,(3,224,224)))
            loss = outputs.loss
            loss.backward()
            
            # log in the loss
            run["train/epoch/loss"].log(loss.data.item())
            # log in the learning rate (lr)
            run["train/epoch/lr"].log(lr_scheduler.get_lr()[0])

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            # evaluate on the evaluation set every X training steps
            epoch_iterator_eval = tqdm(eval_dataloader, desc="Iteration")
            # either model.eval() or model.train(mode=False) to tell that you are testing
            model.eval()
            acc = 0
            em = 0
            f1 = 0
            preds = 0
            if eval_step % config['model_config']['eval_every'] == 0:
                for eval_step, eval_batch in enumerate(epoch_iterator_eval):
                    eval_batch_for_pred = {k: v.to(device) if k not in ["answer_text","golden_answer","id","agent"] else v for k, v in eval_batch.items() } 
                    with torch.no_grad():
                        outputs = model(input_ids=eval_batch_for_pred["input_ids"],labels=eval_batch_for_pred['labels'],attention_mask=eval_batch_for_pred["attention_mask"])

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    
                    logger.info(f"Evaluating on dev set at step {eval_step}")
                    acc_temp, em_temp, f1_temp, preds_temp = evaluate(predictions,eval_batch_for_pred["labels"],eval_batch,agents,logits) 
                    acc += acc_temp
                    em += em_temp
                    f1 += f1_temp
                    preds += preds_temp
                run[f"dev/eval/acc"].log(round(acc*100, 2))
                run[f"dev/eval/em"].log(em)
                run[f"dev/eval/f1"].log(f1)
                run["dev/eval/preds"].log(preds.tolist())

    run.stop()