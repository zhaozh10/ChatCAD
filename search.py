from googlesearch import search
from markdownify import markdownify as md
from typing import List
import time
import requests
import re
import os
# os.environ["http_proxy"]="http://127.0.0.1.1:7890"
# os.environ["https_proxy"]="http://127.0.0.1:7890"

def cleanify(s: str, disease:str) -> str:
    phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
    results = []
    for line in s.split('\n'):
        if len(line) > 1:
            c = line[0]
            if c == "#" or c.isalpha() or c.isdigit():
                if re.search(phone_pattern, line) or "https" in line or (not re.search(disease, line, re.IGNORECASE)):
                    continue
                results.append(line)

    return '\n'.join(results)

def postClean(s:str, disease:str)->str:
    results=[]
    for line in s.split('#'):
        if re.search(disease, line, re.IGNORECASE):
            results.append(line)

    return '\n'.join(results)


def WebSearch(disease: str,web:str,numResults:int):
    res=[]
    query = f"site%3A{web}+{disease}"
    # num_results = numResults

    results = search(query, num_results=numResults)
    # time.sleep(5)
    for result in results:
        # print(result)
        if 'ncbi' in result:
            continue
        response = requests.get(result)
        html = response.content
        md_ = cleanify(md(html),disease)
        md_=postClean(md_,disease)
        res.append(md_)
        # print(md_)
        # print(result)
    return res

if __name__=="__main__":
    webList=['mayoclinic.org','cleveland.org','massgeneral.org','hopkinsmedicine.org','mdanderson.org']
    keyword=["pleural effusion","edema"]
    knowledge="Provided knowledge:\n"
    # assert isinstance(keyword,list)
    for word in keyword:
        numResults=1
        res=WebSearch(word,webList,numResults)
        info=""
        for k in res:
            info+=k+'\n'
        knowledge+=info+'\n'
    print("hold")