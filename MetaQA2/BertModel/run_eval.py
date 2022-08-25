from evaluation import MetaQA2_Eval

data_sets = ["TriviaQA","SearchQA","DuoRC","HotpotQA","NaturalQuestionsShort","NewsQA","QAMR","SQuAD"]

#data_sets =["TriviaQA","SearchQA"]
agents = ["spanbert-large-cased_SearchQA","spanbert-large-cased_TriviaQA-web"]
model_path = "/ukp-storage-1/khammari/Bert10K_shuffledData"

for data_set in data_sets:
    eval_data = "/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/training/nq-nli-eval-TriviaQA+SearchQA-Datasets/nq-nli-eval-triviaQA+SearchQA-"+data_set+".jsonl"
    evaluation_instance = MetaQA2_Eval(model_path,eval_data,agents)
    print("*******************************",data_set,"*******************************")
    evaluation_instance.getConfusionMatrix()
    print(evaluation_instance.get_accuracy())
    print(evaluation_instance.get_f1_EM())