import json
from chat_bot import gpt_bot
import os
from chat_bot import MyEncoder
from query import query_msd
import os
import torch
import sys
from prompt import prob2text,prob2text_zh
from engine_LLM.api import answer_quest
os.environ["http_proxy"]="http://127.0.0.1.1:7890"
os.environ["https_proxy"]="http://127.0.0.1:7890"

if __name__=="__main__":
    # # for web app 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
    # # reserved key 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
    # # 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 $3.8 left
    # # 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
    # chatcad=gpt_bot(engine="gpt-3.5-turbo",api_key="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")
    # testset=json.load(open('../data/mimic-cxr-reports/key/mimic_exp_1k.json'))
    # chatcad.start()
    # chatcad.report_stream(testset,ref=True)

    # chest_base_dict={"肺不张":0,
    #     # "心脏肥大":1, 仅是现象，有好几种可能疾病，需要医生才能判断
    #     "胸腔积液":2,
    #     # "肺浸润":3, 仅是现象，有好几种可能疾病，需要医生才能判断
    #     "纵隔肿块":4,
    #     # "肺结节":5, 默沙东上仅定义了《孤立性肺结节》
    #     # "肺炎":6,
    #     "吸入性肺炎":6.1,"社区获得性肺炎":6.2,"医院获得性肺炎":6.3,"呼吸机相关性肺炎":6.4,"免疫缺陷患者肺炎":6.5,
    #     "气胸":7,
    #     # "肺实变":8, 仅是现象，有好几种可能疾病，需要医生才能判断
    #     "肺水肿":9,
    #     # "肺气肿":10, # 是慢性阻塞性肺病的一种，默沙东上仅对慢性阻塞性肺病进行了定义
    #     "慢性阻塞性肺病（COPD）":10,
    #     "胸膜纤维化和钙化":11,
    #     # "胸膜增厚":12,
    #     "膈疝":13,
    # }
    # key="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"
    # query,knowledge=answer_quest("报告说我有胸腔积液，该怎么办啊？",api_key=key,topic_base_dict=chest_base_dict)
    # print("hold")
    fivedisease_zh={
    "心脏肥大":0,
    "肺水肿":1,
    "肺实变":2,
    "肺不张":3,
    "胸腔积液":4,
        }
    fivedisease={
        "Cardiomegaly":0,
        "Edema":1,
        "Consolidation":2,
        "Atelectasis":3,
        "Pleural Effusion":4,
           }
    info=json.load(open("exp_res.json"))
    info_data=[]
    for elem in info:
        converter=prob2text(torch.tensor(elem['prob']),fivedisease)
        res=converter.promptB().replace("Network A:\n","")
        info_data.append({"gt":elem['gt'],'raw':elem['raw'],"prompt":res,'refined':elem['refined'],})

    with open("./mimic_raw_refine_en.json",'w',encoding='utf-8') as f:
        json.dump(info_data,f,indent=4,ensure_ascii=False)
    print("hold")


