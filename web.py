from glob import glob
import os
import time
import gradio as gr
from chat_bot import gpt_bot
import nibabel as nib
import cv2
from datetime import datetime

# os.environ["http_proxy"]="http://127.0.0.1.1:7890"
os.environ["https_proxy"]="http://127.0.0.1:7890"


title = """<h1 align="center">ChatCAD plus</h1>"""
description = """**这是ChatCAD-plus的早期测试版本，欢迎任何反馈和贡献<br>-将胸片、牙片等图像上传至聊天框，即可获得ChatCAD-plus对该影像的分析<br>-可以继续与ChatCAD-plus交流，进一步了解可能的病症<br>-ChatCAD-plus会在必要的时候给出相关资料的链接**"""
chatbot_bindings =  None
chatbot = None


def chatcad(history):
    if chatbot_bindings is None:
        response = '''**请先输入API key，然后点击保存。**'''
        history[-1][1] = response
        yield history
    else:
        # chat bot put here
        # response = '''**That's cool!**'''
        if isinstance(history[-1][0],str):
            prompt=history[-1][0]
            response = chatbot_bindings.chat(prompt)
        else:
            response = chatbot_bindings.report_zh(history[-1][0]['name'])

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
        cv2.imwrite("./imgs/temp/" + str(update_time) + ".jpg", temp_img)
        img_path = "./imgs/temp/" + str(update_time) + ".jpg"
        
        
    history = history + [((img_path,), None)]
    return history

def add_state(info, history):
    try:
        global chatbot_bindings
        chatbot_bindings = gpt_bot(engine="gpt-3.5-turbo",api_key=info)
        chatbot_bindings.start()
        response = '**初始化成功！**'
    except:
        chatbot_bindings = None
        response = '**初始化失败，请输入正确的openai key。**'
        
    history = history + [(None, response)]
    return history

def clean_data():
    return [(None, description)], None, None


def example_img(i, history):
    return i


callback = gr.CSVLogger()

with gr.Blocks(css="""#col_container1 {margin-left: auto; margin-right: auto;}
                      #col_container2 {margin-left: auto; margin-right: auto;}
                      #chatbot {height: 770px;}
                      #upload_btn {height: auto;}""") as demo:
    gr.HTML(title)
    with gr.Row():
        with gr.Column(scale=0.2):
            with gr.Row():
                #top_p, temperature, top_k, repetition_penalty
                with gr.Accordion("设置", open=True):
                    with gr.Row():
                        api_key_input = gr.Textbox(placeholder="请输入API key", label="API key")
                    with gr.Row():
                        api_key_submit = gr.Button("保存")
            with gr.Row():
                gr.Markdown("### 请上传您想要咨询的医学图像   若您没有图像，可以下载下方的示例图像")
            with gr.Row():
                upload_file = gr.UploadButton("📁上传图像", file_types=["file"], elem_id='upload_btn').style(size='lg')
            with gr.Row():
                img_i = gr.Image(show_label=False, type="numpy", interactive=False)
                gr.Examples(
                    [ 
                        os.path.join(os.path.dirname(__file__), "imgs/examples/chest.jpg"),
                        os.path.join(os.path.dirname(__file__), "imgs/examples/tooth.jpg"),
                    ],
                    img_i,
                    img_i,
                    example_img,
                    label="示例图像"
                )
        with gr.Column(scale=0.8):
            with gr.Row():
                with gr.Column(elem_id = "col_container1"):
                    chatbot = gr.Chatbot(value=[(None, description)], label="ChatCAD plus", elem_id='chatbot').style(height=700) #c
            with gr.Row():
                with gr.Column(elem_id = "col_container2", scale=0.85):
                    inputs = gr.Textbox(label="聊天框", placeholder="请输入文本或者上传图片") #t
                with gr.Column(elem_id = "col_container2", scale=0.15, min_width=0):
                    with gr.Row():
                        inputs_submit = gr.Button("发送", elem_id='inputs_submit')
                    with gr.Row():
                        clean_btn = gr.Button("清空", elem_id='clean_btn')
                    
                
    
    api_key_submit.click(add_state, [api_key_input, chatbot], [chatbot])
    
    inputs_submit.click(add_text, [chatbot, inputs], [chatbot, inputs]).then(
        chatcad, chatbot, chatbot
    )
    
    clean_btn.click(clean_data, [], [chatbot, inputs, img_i])
    
    inputs.submit(add_text, [chatbot, inputs], [chatbot, inputs]).then(
        chatcad, chatbot, chatbot
    )
    
    upload_file.upload(add_file, [chatbot, upload_file], [chatbot]).then(
        chatcad, chatbot, chatbot
    )
    
    # 127.0.0.1.1:7890
    # demo.queue().launch(server_port=4900, server_name="0.0.0.0", favicon_path="shtu.ico")
    demo.queue().launch(server_port=4900, server_name="127.0.0.1", favicon_path="shtu.ico")
    # demo.queue().launch(server_port=7890, server_name="127.0.0.1", favicon_path="shtu.ico")

    
    
