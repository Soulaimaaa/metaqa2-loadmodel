from DataPreparationForNLI import DataPreparationForNLI

agents = ["spanbert-large-cased_SQuAD","spanbert-large-cased_TriviaQA-web"]                     
data_sets =["SQuAD","TriviaQA-web"]

prepare_data = DataPreparationForNLI(agents,data_sets,5000)
prepare_data.writeData("/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/training/Bert5KPerAgent/id-question-5K-spanbert-large-cased_TriviaQA+SQuAD.jsonl")