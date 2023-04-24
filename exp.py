import json
# from chat_bot import gpt_bot
import os
# from chat_bot import MyEncoder
import os
import torch
import sys
from cxr.diagnosis import JFinfer,JFinit,getJFImg
from r2g.report_generate import reportGen
from prompt import prob2text,prob2text_zh
from engine_LLM.api import answer_quest
os.environ["http_proxy"]="http://127.0.0.1.1:7890"
os.environ["https_proxy"]="http://127.0.0.1:7890"

if __name__=="__main__":

    img_model,imgcfg=JFinit('./cxr/config/JF.json','./weights/JFchexpert.pth')

    imgPath='../data/mimic-cxr/p10/p10005866/s55665483/6039e5db-d35aed7c-106102aa-126d200e-a262c646.jpg'
    img1,img2=getJFImg(imgPath,imgcfg) 

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
    generator=reportGen()
    print("hold")

   


