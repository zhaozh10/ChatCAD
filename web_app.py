from flask import Flask, jsonify, request, render_template, Response, stream_with_context, redirect, flash
import argparse
import threading
from io import StringIO, BytesIO
import sys
import base64
import re
import sqlite3
from datetime import datetime
from chat_bot import gpt_bot
import sqlite3
import json
import time 
import traceback
import os
from flask import send_from_directory

os.environ["http_proxy"]="http://127.0.0.1.1:7890"
os.environ["https_proxy"]="http://127.0.0.1:7890"


def has_word(string):
    pattern = re.compile(r'[\u4e00-\u9fa5a-zA-Z]+') # 匹配中文或英文单词
    match = pattern.search(string)
    return match is not None

#=================================== Database ==================================================================
class Discussion:
    def __init__(self, discussion_id, db_path='database.db'):
        self.discussion_id = discussion_id
        self.db_path = db_path

    @staticmethod
    def create_discussion(db_path='database.db', title='untitled'):
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO discussion (title) VALUES (?)", (title,))
            discussion_id = cur.lastrowid
            conn.commit()
        return Discussion(discussion_id, db_path)
    
    @staticmethod
    def get_discussion(db_path='database.db', id=0):
        return Discussion(id, db_path)

    def add_message(self, sender, content):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('INSERT INTO message (sender, content, discussion_id) VALUES (?, ?, ?)',
                         (sender, content, self.discussion_id))
            message_id = cur.lastrowid
            conn.commit()
        return message_id
    @staticmethod
    def get_discussions(db_path):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM discussion')
            rows = cursor.fetchall()
        return [{'id': row[0], 'title': row[1]} for row in rows]

    @staticmethod
    def rename(db_path, discussion_id, title):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE discussion SET title=? WHERE id=?', (title, discussion_id))
            conn.commit()

    def delete_discussion(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('DELETE FROM message WHERE discussion_id=?', (self.discussion_id,))
            cur.execute('DELETE FROM discussion WHERE id=?', (self.discussion_id,))
            conn.commit()

    def get_messages(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('SELECT * FROM message WHERE discussion_id=?', (self.discussion_id,))
            rows = cur.fetchall()
        return [{'sender': row[1], 'content': row[2], 'id':row[0]} for row in rows]
    


    def update_message(self, message_id, new_content):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('UPDATE message SET content = ? WHERE id = ?', (new_content, message_id))
            conn.commit()

    def remove_discussion(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.cursor().execute('DELETE FROM discussion WHERE id=?', (self.discussion_id,))
            conn.commit()

def last_discussion_has_messages(db_path='database.db'):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM message ORDER BY id DESC LIMIT 1")
        last_message = c.fetchone()
    return last_message is not None

def export_to_json(db_path='database.db'):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM discussion')
        discussions = []
        for row in cur.fetchall():
            discussion_id = row[0]
            discussion = {'id': discussion_id, 'messages': []}
            cur.execute('SELECT * FROM message WHERE discussion_id=?', (discussion_id,))
            for message_row in cur.fetchall():
                discussion['messages'].append({'sender': message_row[1], 'content': message_row[2]})
            discussions.append(discussion)
        return discussions

def remove_discussions(db_path='database.db'):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('DELETE FROM message')
        cur.execute('DELETE FROM discussion')
        conn.commit()

# create database schema
def check_discussion_db(db_path):
    print("Checking discussions database...")
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS discussion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS message (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                discussion_id INTEGER NOT NULL,
                FOREIGN KEY (discussion_id) REFERENCES discussion(id)
            )
        ''')
        conn.commit()

    print("Ok")

# ========================================================================================================================


app = Flask("ChatWebUI", static_url_path='/static', static_folder='static')
class ChatWebUI():
    def __init__(self, app, db_path='database.db') -> None:
        self.current_discussion = None
        self.chatbot_bindings = None
        self.app=app
        self.db_path= db_path
        self.add_endpoint('/', '', self.index, methods=['GET','POST'])
        self.add_endpoint('/chat_index', 'chat_index', self.index_chat, methods=['GET'])
        self.add_endpoint('/export', 'export', self.export, methods=['GET'])
        self.add_endpoint('/new_discussion', 'new_discussion', self.new_discussion, methods=['GET'])
        self.add_endpoint('/bot', 'bot', self.bot, methods=['POST'])
        self.add_endpoint('/discussions', 'discussions', self.discussions, methods=['GET'])
        self.add_endpoint('/rename', 'rename', self.rename, methods=['POST'])
        self.add_endpoint('/get_messages', 'get_messages', self.get_messages, methods=['POST'])
        self.add_endpoint('/delete_discussion', 'delete_discussion', self.delete_discussion, methods=['POST'])

        self.add_endpoint('/update_message', 'update_message', self.update_message, methods=['GET'])
        # self.prepare_query(conditionning_message)
        # # chatbot_bindings.generate(conditionning_message, n_predict=55, new_text_callback=self.new_text_callback, n_threads=8)
        # # chatbot_bindings.start()
        # chatbot_bindings.chat(conditionning_message)
        # print(f"Bot said:{self.bot_says}")     

    def prepare_query(self, message):
        self.bot_says=''
        self.full_text=''
        self.is_bot_text_started=False
        self.current_message = message


    # def new_text_callback(self, text: str):
    #     print(text, end="")
    #     self.full_text += text
    #     if self.is_bot_text_started:
    #         self.bot_says += text
    #     if self.current_message in self.full_text:
    #         self.is_bot_text_started=True

    # def new_text_callback_with_yield(self, text: str):
    #     """
    #     To do , fix the problem with yield to be able to show interactive response as text comes
    #     """
    #     print(text, end="")
    #     self.full_text += text
    #     if self.is_bot_text_started:
    #         self.bot_says += text
    #     if self.current_message in self.full_text:
    #         self.is_bot_text_started=True
    #     yield text

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, methods=['GET'], *args, **kwargs):
        self.app.add_url_rule(endpoint, endpoint_name, handler, methods=methods, *args, **kwargs)

    def index_chat(self):
        return render_template('chat.html')
    
    def index(self):
        if request.method == 'GET':
            return render_template('index.html')
        else:
            api_key = request.json['message']
            print(api_key)
            self.chatbot_bindings = gpt_bot(engine="gpt-3.5-turbo",api_key=api_key)
            self.chatbot_bindings.start()
            return jsonify({'message': 'ok'})

    def format_message(self, message):
        # Look for a code block within the message
        pattern = re.compile(r"(```.*?```)", re.DOTALL)
        match = pattern.search(message)

        # If a code block is found, replace it with a <code> tag
        if match:
            code_block = match.group(1)
            message = message.replace(code_block, f"<code>{code_block[3:-3]}</code>")

        # Return the formatted message
        return message

    def export(self):
        return jsonify(export_to_json(self.db_path))

    @stream_with_context
    def parse_to_prompt_stream(self, message, message_id):
        bot_says = ''
        self.stop=False
        
        img = message.split('-')[1]
        message = message.split('-')[0]
        refined_report=None
        if img != 'none':
            # base64编码的图像
            decoded_bytes = base64.b64decode(img.split(',')[1])
            pseudo_path=BytesIO(decoded_bytes)
            refined_report=self.chatbot_bindings.report_zh(pseudo_path,mode='run')
            # yield "这波我并不是很认可你"
            
            # pass

        # send the message to the bot
        print(f"Received message : {message}")
        # First we need to send the new message ID to the client
        # response_id = self.current_discussion.add_message("GPT4All",'') # first the content is empty, but we'll fill it at the end
        response_id = self.current_discussion.add_message("ChatWebUI",'') # first the content is empty, but we'll fill it at the end
        yield(json.dumps({'type':'input_message_infos','message':message, 'id':message_id, 'response_id':response_id}))

        # self.current_message = "User: "+message+"\nChatCAD+:"
        self.current_message = message
        self.prepare_query(self.current_message)
        # chatbot_bindings.generate(self.current_message, n_predict=55, new_text_callback=self.new_text_callback, n_threads=8)
        if has_word(message)==False:
            ret_info=''
        else:
            ret_info=self.chatbot_bindings.chat(self.current_message)
        if refined_report==None:
            self.bot_says=ret_info
        else:
            self.bot_says="诊断报告如下：\n"+refined_report+'\n'+ret_info
        self.current_discussion.update_message(response_id,self.bot_says)
        yield self.bot_says
        # TODO : change this to use the yield version in order to send text word by word

        return "\n".join(bot_says)
            
    def bot(self):
        self.stop=True
        with sqlite3.connect(self.db_path) as conn:
            try:
                if self.current_discussion is None or not last_discussion_has_messages(self.db_path):
                    self.current_discussion=Discussion.create_discussion(self.db_path)

                message = request.json['message']
                text_message = message['text']
                if '-' in text_message:
                    text_message = text_message.replace('-', '_')
                img_message = message['img']
                message_id = self.current_discussion.add_message("user", text_message + '-' + img_message)    

                # Segmented (the user receives the output as it comes)
                # We will first send a json entry that contains the message id and so on, then the text as it goes
                return Response(stream_with_context(self.parse_to_prompt_stream(text_message + '-' + img_message, message_id)))
            except Exception as ex:
                print(ex)
                msg = traceback.print_exc()
                return "<b style='color:red;'>Exception :<b>"+str(ex)+"<br>"+traceback.format_exc()+"<br>Please report exception"
            
    def discussions(self):
        try:
            discussions = Discussion.get_discussions(self.db_path)    
            return jsonify(discussions)
        except Exception as ex:
            print(ex)
            msg = traceback.print_exc()
            return "<b style='color:red;'>Exception :<b>"+str(ex)+"<br>"+traceback.format_exc()+"<br>Please report exception"

    def rename(self):
        data = request.get_json()
        id = data['id']
        title = data['title']
        Discussion.rename(self.db_path, id, title)    
        return "改名成功"
        # return "renamed successfully"

    # switch between different discussions
    def get_messages(self):
        data = request.get_json()
        id = data['id']
        self.current_discussion = Discussion(id,self.db_path)
        messages = self.current_discussion.get_messages()
        return jsonify(messages)
    
    # delete one discussion
    def delete_discussion(self):
        data = request.get_json()
        id = data['id']
        self.current_discussion = Discussion(id, self.db_path)
        self.current_discussion.delete_discussion()
        self.current_discussion = None
        return jsonify({})
    
    def update_message(self):
        try:
            id = request.args.get('id')
            new_message = request.args.get('message')
            self.current_discussion.update_message(id, new_message)
            return jsonify({"status":'ok'})
        except Exception as ex:
            print(ex)
            msg = traceback.print_exc()
            return "<b style='color:red;'>Exception :<b>"+str(ex)+"<br>"+traceback.format_exc()+"<br>Please report exception"

    def new_discussion(self):
        title = request.args.get('title')
        self.current_discussion= Discussion.create_discussion(self.db_path, title)
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # add a new discussion
        self.chatbot_bindings.end()
        self.chatbot_bindings.start()

        # Return a success response
        return json.dumps({'id': self.current_discussion.discussion_id})
    
    
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'shtu.ico')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the chatbot Flask app.')

    parser.add_argument('--temp', type=float, default=0.1, help='Temperature parameter for the model.')
    parser.add_argument('--n_predict', type=int, default=128, help='Number of tokens to predict at each step.')
    parser.add_argument('--top_k', type=int, default=40, help='Value for the top-k sampling.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Value for the top-p sampling.')
    parser.add_argument('--repeat_penalty', type=float, default=1.3, help='Penalty for repeated tokens.')
    parser.add_argument('--repeat_last_n', type=int, default=64, help='Number of previous tokens to consider for the repeat penalty.')
    parser.add_argument('--ctx_size', type=int, default=2048, help='Size of the context window for the model.')
    parser.add_argument('--debug',dest='debug', action='store_true', help='launch Flask server in debug mode')
    parser.add_argument('--host', type=str, default='localhost', help='the hostname to listen on')
    parser.add_argument('--port', type=int, default=9600, help='the port to listen on')
    parser.add_argument('--db_path', type=str, default='database.db', help='Database path')
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    # chatbot_bindings=gpt_bot(engine="gpt-3.5-turbo",api_key="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")
    # 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
    # chatbot_bindings = Model(ggml_model='./models/gpt4all-converted.bin', n_ctx=512)
    # app.config['UPLOAD_FOLDER'] = 'static/uploads'
    
    check_discussion_db(args.db_path)
    # chatbot_bindings.start()
    bot = ChatWebUI(app, args.db_path)
    # bot = Gpt4AllWebUI(chatbot_bindings, app, args.db_path)

    # if args.debug:
    if True:
        app.run(debug=True, port=4900)
        # app.run(debug=True, host=args.host, port=args.port)
    else:
        app.run(host=args.host, port=args.port)
