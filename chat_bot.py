import re
from revChatGPT.V3 import Chatbot
from transformers import pipeline
import time
from text2vec import SentenceModel
import torch
from googlesearch import search
from search import WebSearch
from prompt import prob2text,prob2text_zh
from report_generate import getJFImg,JFinfer,JFinit,reportGen
import json
import os
from tqdm import tqdm
from query import query_msd, query_prompt
from engine_LLM.api import answer_quest,query_range
from modality_identify import ModalityClip
from dental_eval import  PAN,Score2txt



class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
            return obj
        return super().default(obj)

dental_config = {
        'detection': './weights/detection.pt',
        'segmentation': './weights/segmentation.pt',
        'classification': './weights/classification.pt',
        'detection_threshold':0.2,
        'crop_area': (384, 256)
    }


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

chest_base_dict={"肺不张":0,
        # "心脏肥大":1, 仅是现象，有好几种可能疾病，需要医生才能判断
        "胸腔积液":2,
        # "肺浸润":3, 仅是现象，有好几种可能疾病，需要医生才能判断
        "纵隔肿块":4,
        # "肺结节":5, 默沙东上仅定义了《孤立性肺结节》
        # "肺炎":6,
        "吸入性肺炎":6.1,"社区获得性肺炎":6.2,"医院获得性肺炎":6.3,"呼吸机相关性肺炎":6.4,"免疫缺陷患者肺炎":6.5,
        "气胸":7,
        # "肺实变":8, 仅是现象，有好几种可能疾病，需要医生才能判断
        "肺水肿":9,
        # "肺气肿":10, # 是慢性阻塞性肺病的一种，默沙东上仅对慢性阻塞性肺病进行了定义
        "慢性阻塞性肺病（COPD）":10,
        "胸膜纤维化和钙化":11,
        # "胸膜增厚":12,
        "膈疝":13,
        "牙周炎":14,
        "牙龈炎":15,
        "龋齿":16,
        "牙髓炎":17
    }


def ret_website(query:str,online: bool=False):
    if online:
        keyword = f"site:https://www.msdmanuals.cn/professional {query}"
        website = next(search(keyword, num=1, stop=1, pause=5))
        return website
    else:
        info={
            "肺不张":"https://www.msdmanuals.cn/professional/pulmonary-disorders/bronchiectasis-and-atelectasis/atelectasis",
            "胸腔积液":"https://www.msdmanuals.cn/professional/pulmonary-disorders/mediastinal-and-pleural-disorders/pleural-effusion",
            "纵膈肿块":"https://www.msdmanuals.cn/professional/pulmonary-disorders/mediastinal-and-pleural-disorders/mediastinal-masses",
            "胸膜纤维化和钙化":"https://www.msdmanuals.cn/professional/pulmonary-disorders/mediastinal-and-pleural-disorders/pleural-fibrosis-and-calcification",
            "气胸":"https://www.msdmanuals.cn/professional/pulmonary-disorders/mediastinal-and-pleural-disorders/pneumothorax",
            "肺水肿":"https://www.msdmanuals.cn/professional/cardiovascular-disorders/heart-failure/pulmonary-edema",
            "慢性阻塞性肺病（COPD）":"https://www.msdmanuals.cn/professional/pulmonary-disorders/chronic-obstructive-pulmonary-disease-and-related-disorders/chronic-obstructive-pulmonary-disease-copd",
            "膈疝":"https://www.msdmanuals.cn/professional/pediatrics/congenital-gastrointestinal-anomalies/diaphragmatic-hernia",
            "吸入性肺炎":"https://www.msdmanuals.cn/professional/pulmonary-disorders/pneumonia/aspiration-pneumonitis-and-pneumonia",
            "社区获得性肺炎":"https://www.msdmanuals.cn/professional/pulmonary-disorders/pneumonia/community-acquired-pneumonia",
            "医院获得性肺炎":"https://www.msdmanuals.cn/professional/pulmonary-disorders/pneumonia/hospital-acquired-pneumonia",
            "呼吸机相关性肺炎":"https://www.msdmanuals.cn/professional/pulmonary-disorders/pneumonia/ventilator-associated-pneumonia",
            "免疫缺陷患者肺炎":"https://www.msdmanuals.cn/professional/pulmonary-disorders/pneumonia/pneumonia-in-immunocompromised-patients",
            "牙周炎":"https://www.msdmanuals.cn/professional/dental-disorders/periodontal-disorders/periodontitis",
            "牙龈炎":"https://www.msdmanuals.cn/professional/dental-disorders/periodontal-disorders/gingivitis",
            "龋齿":"https://www.msdmanuals.cn/professional/dental-disorders/common-dental-disorders/caries",
            "牙髓炎":"https://www.msdmanuals.cn/professional/dental-disorders/common-dental-disorders/pulpitis"
        }
        if query in info.keys():
            return info[query]
        else:
            keyword = f"site:https://www.msdmanuals.cn/professional {query}"
            website = next(search(keyword, num=1, stop=1, pause=5))
            return website



class base_bot:
    def start(self):
        """为当前会话新建chatbot"""
        pass
    def reset(self):
        """删除当前会话的chatbot"""
        pass
    def chat(self,message: str):
        pass


class gpt_bot(base_bot):
    def __init__(self, engine: str,api_key: str,):
        """初始化模型"""
        self.agent=None
        self.engine=engine
        self.api_key=api_key
        # self.agent = Chatbot(engine=engine,api_key=api_key)
        img_model,imgcfg=JFinit('./config/JF.json','../preTrain/JFchexpert.pth')
        self.imgcfg=imgcfg
        self.img_model=img_model
        self.reporter=reportGen()
        self.modality=["chest x-ray", "panoramic dental x-ray", "knee mri","Mammography"]
        self.translator = pipeline(model="zhaozh/radiology-report-en-zh-ft-base",device=0,max_length=500,truncation=True)
        self.identifier=ModalityClip(self.modality)
        self.dental_net = PAN(dental_config)
        self.sent_model = SentenceModel()
    
        

    def radio_en_zh(self, content: str):
        output=self.translator(content)
        report_zh=output[0]['translation_text']
        return report_zh

    def chat_with_gpt(self, prompt):

        while True:
            try:
                message=self.agent.ask(prompt)
            except:
                # repeat+=1
                time.sleep(2)
                continue
            break
        return message
    
    def start(self):
        """为当前会话新建chatbot"""
        if self.agent is not None:
            self.agent.reset()
        self.agent = Chatbot(engine=self.engine,api_key=self.api_key)
        # You should be able to recommend conventional medicines, treatments and other natural alternatives.
        instruction="Act as a doctor named ChatCAD-plus. All your answers should be in Chinese."
        self.chat_with_gpt(instruction)
        return 

        # pass

    def reset(self):
        if self.agent is not None:
            return
        else:
            self.agent.reset()
            return

    def end(self):
        """删除当前会话的chatbot"""
        if self.agent is not None:
            self.agent=None
        else:
            print("No remaining agent to be ended!")


    def report_en(self,img_path,highway: dict=None, ref: bool=False,mode="run"):
        if highway is not None:
            text_report=highway['raw']
            prob=torch.tensor(highway['prob'])
            # pass
        else:
            img1,img2=getJFImg(img_path,self.imgcfg) 
            text_report=self.reporter.report(img1)[0]
            prob=JFinfer(self.img_model,img2,self.imgcfg)
        awesomePrompt="\nRevise the report of Network B based on Network A." 
        ref_prompt=""
        if ref:
            ref_list=query_msd(text_report,k=3)
            ref_prompt=query_prompt(ref_list)
            ref_prompt=f"following the style of these examples:\n{ref_prompt}"
            # awesomePrompt=f"\nRevise the report of Network B based on Network A following the style of these examples:\n{ref_prompt}" 
            awesomePrompt=f"\nRevise the report of Network B based on Network A." 

        
        converter=prob2text(prob,fivedisease)
        res=converter.promptB()
        prompt_report=res+"Network B generate a report："+text_report
        constraints="Strictly follow these constraints: Please do not mention Network A, Network B, and any provided results. Suppose you are a doctor writing findings for a chest x-ray report."
        if ref_prompt!="":
            prompt_report=prompt_report+awesomePrompt+" "+constraints[:-1]+" "+ref_prompt
        else:
            prompt_report=prompt_report+awesomePrompt+" "+constraints+ref_prompt
        refined_report = self.chat_with_gpt(prompt_report)
        if mode=="debug":
            return text_report,refined_report,prob.detach().cpu().numpy().tolist()
        else:
            return refined_report

    def report_cxr_zh(self,img_path,highway: bool=False, mode:str='run'):
        if highway:
            pass
        else:
            img1,img2=getJFImg(img_path,self.imgcfg) 
            text_report=self.reporter.report(img1)[0]
            
            text_report=self.radio_en_zh(text_report)
            prob=JFinfer(self.img_model,img2,self.imgcfg)
        converter=prob2text_zh(prob,fivedisease_zh)
        # converter=prob2text(prob,fivedisease)
        res=converter.promptB()
        prompt_report_zh=res+" 网络B生成了诊断报告："+text_report
        awesomePrompt_zh="\nRefine the report of Network B based on results from Network A using Chinese.Please do not mention Network A and \
            Network B. Suppose you are a doctor writing findings for a chest x-ray report."
        prompt_report_zh=prompt_report_zh+awesomePrompt_zh
        refined_report = self.chat_with_gpt(prompt_report_zh)

        if mode=='debug':
            return text_report,refined_report,prob.detach().cpu().numpy().tolist()
        else:
            return refined_report

    def report_dental_zh(self,img_path):
        txt_prompt = Score2txt(self.dental_net.excution(img_path)).promptGeneration()
        return txt_prompt

    def report_zh(self,img_path,highway: bool=False, mode:str='run'):
        index=self.identifier.identify(img_path)
        if index==0:
            return self.report_cxr_zh(img_path,highway,mode)
        if index==1:
            return self.report_dental_zh(img_path)
        else:
            print("error!")
            return
        

    def report_stream(self, testset: list,prefix:str='../data/mimic-cxr/test/', save_step:int=50, ref: bool=False,mode: str='en'):
        res=[]
        highway_info=json.load(open('./exp_res.json'))
        mapping=json.load(open('../data/mimic-cxr-reports/mimic_test_dict.json'))
        info=json.load(open('../data/mimic-cxr-reports/report_dict.json'))
        for i, elem in enumerate(tqdm(testset)):
            self.agent.reset()
            if mode=='en':
                text_report,refined_report,prob=self.report_en(os.path.join(prefix,elem['img']),highway=highway_info[i],ref=ref,mode='debug')
            else:
                text_report,refined_report,prob=self.report(os.path.join(prefix,elem['img']),mode='debug')
            gt=info[mapping[elem['img']]['text_path']]
            res.append({"gt":gt,"raw":text_report,"refined":refined_report,"prob":prob})
      



    def chat(self,message: str,verbose: bool=False):
        check_prompt="用户的以下输入是在针对医学问题进行提问吗？如果是，请回复1，如果不是或者不确定，请回复0，回答的长度为1，请不要有多余的回复。"
        check_message=self.chat_with_gpt(check_prompt+'\n'+message)
        print(f"check message: {check_message}")
        # check_message="0"
        numbers = re.findall(r'\d+', check_message)
        assert len(numbers)==1
        check=eval(numbers[0])
        if check==0:
            return self.chat_with_gpt(message)
        
        
        # refine_prompt="请根据以下内容概括患者的提问：\n"
        refine_prompt="请根据以下内容概括患者的提问并对所涉及的疾病指出其全称：\n"
        refined_message=self.chat_with_gpt(refine_prompt+message)
        topic_range=query_range(self.sent_model,refined_message,k=5)
        
        # ret=answer_quest(refined_message,api_key=self.api_key,topic_base_dict=chest_base_dict)
        ret=answer_quest(refined_message,api_key=self.api_key,topic_base_dict=topic_range)
        # ret=None
        if ret==None:
            response = self.chat_with_gpt(refined_message)
            response +="<br>注：未在默沙东数据库中得到明确依据，请谨慎采纳"
            message=response
        else:
            query,knowledge=ret
            knowledge=knowledge.replace("\n\n","\n")
            needed_site=ret_website(query)
            
            # chat_message="病人："+message+'\n'+refined_message+"\n请在独立思考的基础上将以下知识作为参考来解答病人的问题并给出分析，切记不要机械地重复以下知识：\n"+knowledge
            try:
                index = knowledge.index("：")
            except ValueError:
                index = -1
            knowledge = knowledge[index+1:]
            # print(knowledge)
            chat_message=f"\n请参考以下知识来解答病人的问题“{message}”并给出分析，切记不要机械地重复以下知识：\n"+knowledge
            response = self.chat_with_gpt(chat_message)
            message = response+f"<br><br>注：相关资料来自默沙东医疗手册专业版 <a href={needed_site}, class='underline',style='#c82423' >{query}</a>"
            # message+=f"<br>\"\"\"{knowledge}<br>\"\"\""
        return message


class glm_bot(base_bot):
    def __init__(self, model_path):
        """初始化 GPT 模型"""
        self.model_path = model_path
        # TODO: 在此处加载模型
    
    def start(self):
        """为当前会话新建chatbot"""
        pass

    def end(self):
        """删除当前会话的chatbot"""
        pass

    def chat(self,message: str):
        pass

