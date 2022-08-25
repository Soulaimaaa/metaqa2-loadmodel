from MetaQA2_UpperBoundEvaluation import MetaQA2_UpperBoundEvaluation
data_sets = ["NewsQA"]
agents = ["spanbert-large-cased_HotpotQA","spanbert-large-cased_NaturalQuestionsShort","spanbert-large-cased_NewsQA","spanbert-large-cased_QAMR","spanbert-large-cased_SearchQA","spanbert-large-cased_SQuAD","spanbert-large-cased_TriviaQA-web","spanbert-large-cased_DuoRC"]                     

for data_set in data_sets:
    evaluation_instance = MetaQA2_UpperBoundEvaluation(data_set,agents)
    print(evaluation_instance.evaluate())