import json

class DataPreparationForNLI():
    def __init__(self,agents,data_sets,size):
        self.agents = agents
        self.data_sets = data_sets
        self.size = size
        self.differentids = []
        self.balancedData = []

    def get_data_from_agent(self,agent):
        data_path = "/ukp-storage-1/khammari/QA-Verification-Via-NLI/MetaQA2/evaluation/FullSet/id-question-"+agent+".jsonl"
        with open(data_path) as f:
            data = list(f)
        json_data = [json.loads(d) for d in data]
        for dataset in self.data_sets:
            labelTrue = list(filter(lambda x: x['label'] and x['dataset']==dataset and (x['id'] not in self.differentids), json_data))[:int(self.size/4)]
            labelFalse = list(filter(lambda x: (not x['label']) and x['dataset']==dataset and (x['id'] not in self.differentids), json_data))[:int(self.size/4)]
            self.balancedData.extend(labelTrue)
            self.balancedData.extend(labelFalse)
            self.differentids.extend([ l["id"] for l in labelTrue])
            self.differentids.extend([ l["id"] for l in labelFalse])
    
    def writeData(self,path):   
        for agent in self.agents:  
            self.get_data_from_agent(agent)
        with open(path, 'w') as f:
            for i in range(len(self.balancedData)):
                f.write(json.dumps(self.balancedData[i]) + "\n")

