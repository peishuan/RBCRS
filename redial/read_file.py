import pandas as  pd
import json

#read data in dataframe
# data = pd.read_json("./redial/train_data.jsonl",lines=True)
# print(data['movieMentions'][0])

#read data in dictionary
dataList = []
with open('./redial/train_data.jsonl','r') as f:
    for jsonObj in f:
        dataDic = json.loads(jsonObj)
        dataList.append(dataDic)

# print(dataList[0].get("movieMentions").keys())
# print(dataList[0].get("movieMentions").values())
