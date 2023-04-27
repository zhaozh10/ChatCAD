import os
import time
import gradio as gr
from chat_bot import gpt_bot
import nibabel as nib
import cv2
from datetime import datetime

# os.environ["http_proxy"]="http://127.0.0.1.1:7890"
# os.environ["https_proxy"]="http://127.0.0.1:7890"


title = """<h1 align="center">ChatCAD plus</h1>"""
description = """**这是ChatCAD-plus的早期测试版本，欢迎任何反馈和贡献<br>-将胸片、牙片等图像上传至聊天框，即可获得ChatCAD-plus对该影像的分析<br>-可以继续与ChatCAD-plus交流，进一步了解可能的病症<br>-ChatCAD-plus会在必要的时候给出相关资料的链接**"""
chatbot_bindings =  None


def chatcad(history):
    if chatbot_bindings is None:
        response = '''**请先输入API key，然后点击保存。**'''
        history[-1][1] = response
        yield history
    else:
        # chat bot put here
        response = '''**That's cool!**'''
        history[-1][1] = response
        time.sleep(2)
        yield history

def add_text(history, text):
    history = history + [(text, None)]
    return history, None

def add_file(history, file):
    # This is file path
    print(file.name)
    img_path = file.name
    update_time = str(datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")
    if file.name.endswith(".nii.gz"):
        img = nib.load(img_path)
        _, _, queue = img.dataobj.shape
        temp_img = img.dataobj[:, :, queue//2].T
        cv2.imwrite("imgs/temp/" + str(update_time) + ".jpg", temp_img)
        img_path = "imgs/temp/" + str(update_time) + ".jpg"
        
        
    history = history + [((img_path,), None)]
    return history

def add_state(info, history):
    try:
        chatbot_bindings = gpt_bot(engine="gpt-3.5-turbo",api_key=info)
        chatbot_bindings.start()
        response = '**初始化成功！**'
    except:
        chatbot_bindings = None
        response = '**初始化失败，请输入正确的openai key。**'
        
    history = history + [(None, response)]
    return history


callback = gr.CSVLogger()

with gr.Blocks(css="""#col_container1 {margin-left: auto; margin-right: auto;}
                      #col_container2 {margin-left: auto; margin-right: auto;}
                      #chatbot {height: 770px;}
                      #upload_btn {height: auto;}""") as demo:
    gr.HTML(title)

    with gr.Row():
        with gr.Column(elem_id = "col_container1"):
            chatbot = gr.Chatbot(value=[(None, description)], label="ChatCAD plus", elem_id='chatbot').style(height=700) #c
    with gr.Row():
        with gr.Column(elem_id = "col_container2", scale=0.85):
            inputs = gr.Textbox(label="聊天框", placeholder="请输入文本或者上传图片") #t
        with gr.Column(elem_id = "col_container2", scale=0.15, min_width=0):
            with gr.Row():
                btn = gr.UploadButton("📁", file_types=["file"], elem_id='upload_btn')
            with gr.Row():
                inputs_submit = gr.Button("发送", elem_id='inputs_submit')
            
    with gr.Row():
        #top_p, temperature, top_k, repetition_penalty
        with gr.Accordion("设置", open=True):
            with gr.Row():
                api_key_input = gr.Textbox(placeholder="请输入API key", label="API key")
                api_key_submit = gr.Button("保存")
                
    
    api_key_submit.click(add_state, [api_key_input, chatbot], [chatbot])
    inputs_submit.click(add_text, [chatbot, inputs], [chatbot, inputs]).then(
        chatcad, chatbot, chatbot
    )
    inputs.submit(add_text, [chatbot, inputs], [chatbot, inputs]).then(
        chatcad, chatbot, chatbot
    )
    btn.upload(add_file, [chatbot, btn], [chatbot]).then(
        chatcad, chatbot, chatbot
    )
    
    
    demo.queue().launch(server_port=4900, server_name="0.0.0.0", favicon_path="shtu.ico")
    
