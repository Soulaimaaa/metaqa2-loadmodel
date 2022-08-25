from transformers import DefaultDataCollator, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset,load_metric

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
predictions_path = "/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/training/Bert5KPerAgent/nq-nli-5K-triviaQA+SearchQA.jsonl"
eval_path ="/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/training/nq-nli-eval-TriviaQA+SearchQA-Datasets/nq-nli-eval-triviaQA+SearchQA-TriviaQA.jsonl"
dataset = load_dataset('json', data_files=predictions_path)
evalDataset = load_dataset('json', data_files=eval_path)

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

tokenized_input = dataset.map(preprocess_function, batched=True, remove_columns=['id', 'question_text', 'answer_text', 'agent','golden_answer','context', 'dataset', 'question_statement_text'])
tokenized_eval_input = evalDataset.map(preprocess_function, batched=True, remove_columns=['id', 'question_text', 'answer_text', 'agent', 'golden_answer', 'context', 'dataset', 'question_statement_text'])

data_collator = DefaultDataCollator()
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    #learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    #weight_decay=0.01,
)
print(training_args)
###############################
import numpy as np

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_input['train'],
    eval_dataset=tokenized_eval_input['train'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()

trainer.save_model("./BertModelNLI_SearchQA_Trivia_4K_training_args_changed")



