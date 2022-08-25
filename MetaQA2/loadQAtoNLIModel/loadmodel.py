import json
import logging
from typing import Dict, List

import pandas as pd
from datasets import load_dataset
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer , DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO, filename="/ukp-storage-1/khammari/nq-nli-5K-triviaQA+SearchQA.txt", filemode='w', format='%(message)s')
logging.info(torch.cuda.is_available())
input_file="/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/training/Bert5KPerAgent/id-question-5K-spanbert-large-cased_TriviaQA+SQuAD.jsonl"
output_path='/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/training/Bert5KPerAgent/nq-nli-5K-triviaQA+SQuAD.jsonl'
#############################################################################
# def read_data(data_path, columns):
#     tsv_data = pd.read_csv(data_path, sep='\t', usecols=columns).head(5)
#     result = [tsv_data.iloc[i, 0:3].to_dict() for i in range(len(tsv_data))]
#     return result
# #array of json
# data_train = read_data('./QA-Verification-Via-NLI/dev.tsv',['question', 'answer'])[:5]
# expected_result = read_data('./QA-Verification-Via-NLI/dev.tsv',['rule-based'])[:5]
# print(data_train)
#############################################################################
converter_extraction_path='./QA-Verification-Via-NLI/model_data/question-converter-t5-3b'
config_name = None
cache_dir = None
use_auth_token = False
model_revision = "main"
tokenizer_name = None
use_fast_tokenizer = True
config = AutoConfig.from_pretrained(
        config_name if config_name else converter_extraction_path,
        cache_dir=cache_dir,
        revision=model_revision,
        use_auth_token=True if use_auth_token else None,
    )
tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else converter_extraction_path,
        cache_dir=cache_dir,
        use_fast=use_fast_tokenizer,
        revision=model_revision,
        use_auth_token=True if use_auth_token else None,
    )
model = AutoModelForSeq2SeqLM.from_pretrained(
    converter_extraction_path,
    from_tf=bool(".ckpt" in converter_extraction_path),
    config=config,
    cache_dir=cache_dir,
    revision=model_revision,
    use_auth_token=True if use_auth_token else None,
)
def process_question_converter_qanli(examples: Dict):
    inputs = []
    targets = []
    for a, q in zip(examples['answer_text'],
                    examples['question_text']):
        if a and q:
            inputs.append(
                "{} </s> {}".format(q, a)
            )
            targets.append('DUMB LABEL')
    return inputs, targets
max_source_length =128
max_target_length=128
ignore_pad_token_for_loss = True
padding = False
def preprocess_function(examples):
        inputs = []
        targets = []
        inputs, targets = process_question_converter_qanli(examples)        
        model_inputs = tokenizer(text=inputs,
                                 max_length=max_source_length,
                                 padding=padding,
                                 truncation=True,
                                 add_special_tokens=True
                                 )
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets,
                               max_length=max_target_length,
                               padding=padding,
                               truncation=True
                               )
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
preprocessing_num_workers = None
overwrite_cache = True
raw_prediction_dataset = load_dataset(
            'json',
            data_files=input_file,
            split="train"
        )
prediction_dataset = raw_prediction_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=["label"],
            load_from_cache_file=not overwrite_cache,
            batch_size=50
        )
beam_size = 5
training_args = Seq2SeqTrainingArguments(
            adafactor=False,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-08,
            data_seed=None,
            dataloader_drop_last=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            ddp_bucket_cap_mb=None,
            ddp_find_unused_parameters=None,
            debug=[],
            deepspeed=None,
            disable_tqdm=False,
            do_eval=False,
            do_predict=True,
            do_train=False,
            eval_accumulation_steps=None,
            eval_delay=0,
            eval_steps=None,
            evaluation_strategy="no",
            fp16=False,
            fp16_backend="auto",
            fp16_full_eval=False,
            fp16_opt_level="O1",
            fsdp=[],
            fsdp_min_num_params=0,
            full_determinism=False,
            generation_max_length=None,
            generation_num_beams=None,
            gradient_accumulation_steps=1,
            gradient_checkpointing=False,
            greater_is_better=None,
            group_by_length=False,
            half_precision_backend="auto",
            hub_model_id=None,
            hub_private_repo=False,
            hub_strategy="every_save",
            hub_token="<HUB_TOKEN>",
            ignore_data_skip=False,
            include_inputs_for_metrics=False,
            label_names=None,
            label_smoothing_factor=0.0,
            learning_rate=5e-05,
            length_column_name="length",
            load_best_model_at_end=False,
            log_on_each_node=True,
            logging_dir="./QA-Verification-Via-NLI/model_data/question-converter-t5-3b/runs/May29_18-13-24_blubella",
            logging_first_step=False,
            logging_nan_inf_filter=True,
            logging_steps=500,
            logging_strategy="steps",
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            max_steps=-1,
            metric_for_best_model=None,
            no_cuda=False,
            num_train_epochs=3.0,
            optim="adamw_hf",
            output_dir="./QA-Verification-Via-NLI/model_data/question-converter-t5-3b",
            overwrite_output_dir=True,
            past_index=-1,
            per_device_eval_batch_size=2,
            per_device_train_batch_size=2,
            predict_with_generate=True,
            prediction_loss_only=False,
            push_to_hub=False,
            push_to_hub_model_id=None,
            push_to_hub_organization=None,
            push_to_hub_token="<PUSH_TO_HUB_TOKEN>",
            remove_unused_columns=True,
            report_to=['tensorboard'],
            resume_from_checkpoint=None,
            run_name="./QA-Verification-Via-NLI/model_data/question-converter-t5-3b",
            save_on_each_node=False,
            save_steps=500,
            save_strategy="steps",
            save_total_limit=None,
            seed=42,
            sharded_ddp=[],
            skip_memory_metrics=True,
            sortish_sampler=False,
            tf32=None,
            tpu_metrics_debug=False,
            tpu_num_cores=None,
            use_legacy_prediction_loop=False,
            warmup_ratio=0.0,
            warmup_steps=0,
            weight_decay=0.0,
            xpu_backend=None,
            )

label_pad_token_id = -100 
data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

trainer = Seq2SeqTrainer(
            model=model.to(torch.device('cuda')),
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=None
        )
logging.info('before trainer prediction...')
results = trainer.predict(test_dataset=prediction_dataset,
                                  num_beams= beam_size,
                                  max_length= max_target_length)
predictions, label_ids, metrics = results[0], results[1], results[2]
decoded_preds = tokenizer.batch_decode(predictions,
                                               skip_special_tokens=True)

def write_question_converter_predictions_out(dataset,
                                             predictions: List[str],
                                             output_path: str,
                                             output_format: str = None,
                                             data_source: str = None
                                             ):
    if output_format == 'csv':
        csv_file = open(output_path, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        if data_source == 'qa-nli':
            csv_fields = ['example_id', 'question', 'answer', 'question_statement']
            csv_writer.writerow(csv_fields)
            for data, pred in zip(dataset, predictions):
                csv_writer.writerow([data['example_id'],
                                     data['question_text'],
                                     data['answer_text'],
                                     pred])
        else:
            csv_fields = ['question', 'answer', 'question_statement', 'turker_answer']
            csv_writer.writerow(csv_fields)
            for data, pred in zip(dataset, predictions):
                csv_writer.writerow([data['question'],
                                     data['answer'],
                                     pred,
                                     data['turker_answer']])
    else:
        with open(output_path, 'w') as fout:
            for data, pred in zip(dataset, predictions):
                data['question_statement_text'] = pred
                json.dump(data, fout)
                fout.write('\n')
logging.info('after trainer prediction...')
write_question_converter_predictions_out(
                raw_prediction_dataset,
                decoded_preds,
                output_path,
                output_format='json',
                data_source='qa-nli'
            )
'''input_ids = tokenizer("Who is the president of USA?", return_tensors="pt").input_ids
sequence_ids = model.generate(input_ids)
sequences = tokenizer.batch_decode(sequence_ids)
print(sequences)'''
