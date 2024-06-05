"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
import re
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')
from tqdm import tqdm
import json
#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")

    parser.add_argument("--split_index", type=int, required=False, default=None)

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================
    
# cmd = "--cfg-path eval_configs/video_llama_eval_withaudio.yaml --model_type llama_v2 --gpu-id 1"

# import sys
# sys.argv += cmd.split()

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []

    if img_list is not None:
        img_list = []
    return chat_state, img_list

def gradio_reset_bak(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_imgorvideo(video_path, img_path, text_input, chat_state,chatbot,audio_flag):
    if args.model_type == 'vicuna':
        chat_state = default_conversation.copy()
    else:
        chat_state = conv_llava_llama_2.copy()


    if img_path is None and video_path is None:
        return None, chat_state, None
    elif img_path is not None and video_path is None:
        print(img_path)
        chatbot = chatbot + [((img_path,), None)]
        chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        img_list = [] 
        llm_message = chat.upload_img(img_path, chat_state, img_list)

        # print(img_list)
        return chat_state, img_list, chatbot
    
    elif video_path is not None and img_path is None:
        print(video_path)
        chatbot = chatbot + [((img_path,), None)]
        chat_state.system = ""
        img_list = []
        if audio_flag:
            llm_message = chat.upload_video(video_path, chat_state, img_list)
        else:
            llm_message = chat.upload_video_without_audio(video_path, chat_state, img_list)
        return chat_state, img_list, chatbot
    else:
        chat_state.system = "Currently, only one input is supported"
        return chat_state, None, chatbot



def upload_imgorvideo_bak(gr_video, gr_img, text_input, chat_state,chatbot,audio_flag):
    if args.model_type == 'vicuna':
        chat_state = default_conversation.copy()
    else:
        chat_state = conv_llava_llama_2.copy()
    if gr_img is None and gr_video is None:
        return None, None, None, gr.update(interactive=True), chat_state, None
    elif gr_img is not None and gr_video is None:
        print(gr_img)
        chatbot = chatbot + [((gr_img,), None)]
        chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list,chatbot
    elif gr_video is not None and gr_img is None:
        print(gr_video)
        chatbot = chatbot + [((gr_video,), None)]
        chat_state.system =  ""
        img_list = []
        if audio_flag:
            llm_message = chat.upload_video(gr_video, chat_state, img_list)
        else:
            llm_message = chat.upload_video_without_audio(gr_video, chat_state, img_list)
        return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list,chatbot
    else:
        # img_list = []
        return gr.update(interactive=False), gr.update(interactive=False, placeholder='Currently, only one input is supported'), gr.update(value="Currently, only one input is supported", interactive=False), chat_state, None,chatbot
    
def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        error_message = "Input should not be empty!"
        return error_message, chatbot, chat_state
    
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_ask_bak(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    print(chat_state.get_prompt())
    print(chat_state)
    return chatbot, chat_state, img_list

def numerical_sort_key(filename):
    match = re.search(r'(\d+)(?=\.\w+$)', filename)
    if match:
        return int(match.group(1))
    return float('inf')

# video_path = "examples/boat.mp4"
video_dir = "video_data"
# video_path = "examples/dia0_utt0_0.mp4"
# image_path = None
text_input1 = "what is the emotion state of the speaker? "
text_input2 = "What life distress might explain the speaker’s emotional expression in this picture, summary in ten words."


df = pd.read_csv('data/results_Intreatment.csv')
# json_fpath = os.path.join(data_base_dir, "Intreatment_llama.json")

output_data_dir = "emotion_data"

num_beams = 2

temperature = 0.8

audio = True
results = []
chat_state = []
chatbot = []

# part_results = [] 
filenames = os.listdir(video_dir)

first_write = True
processed = 0

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):

    if args.split_index is not None:
        if index % 2 != args.split_index:
            continue



    print("processing row {}".format(index))

    speaker = row['Speaker']
    utterance_id = row['Utterance_ID']
    dialogue_id = row['Dialogue_ID']
    emotion = row['Emotion']
    strategy = row['Strategy']
    text = row['Utterance']

    file_id = "dia{}_utt{}_{}".format(dialogue_id,utterance_id,index)

    result_fname = "{}.json".format(file_id)
    result_fpath = os.path.join(output_data_dir, result_fname)

    if os.path.exists(result_fpath):
        print("skip existing id: ", file_id)
        continue

    filename = "{}.mp4".format(file_id)
    video_path = os.path.join(video_dir, filename)
    image_path = None
    print("video_path is:",video_path)

    if not os.path.exists(video_path):
        print(f"File not found, skipping: {video_path}")
        break

    # if speaker == 'Client':
    #     chat_state, img_list, chatbot = upload_imgorvideo(video_path, image_path, text_input1, chat_state, chatbot, audio)
    #     _, chatbot, chat_state = gradio_ask(text_input1, chatbot, chat_state)
    #     chatbot, chat_state, img_list = gradio_answer(chatbot, chat_state, img_list, num_beams, temperature)

    #     # _, chatbot, chat_state = gradio_ask(text_input2, chatbot, chat_state)
    #     # chatbot, chat_state, img_list = gradio_answer(chatbot, chat_state, img_list, num_beams, temperature)
    #     print("\nchat_state is\n",chat_state)

    # # else:
    # #     chat_state, img_list, chatbot = upload_imgorvideo(video_path, image_path, text_input1, chat_state, chatbot, audio)
    # #     _, chatbot, chat_state = gradio_ask(text_input1, chatbot, chat_state)
    # #     chatbot, chat_state, img_list = gradio_answer(chatbot, chat_state, img_list, num_beams, temperature)
    # #     print("\nchat_state is\n",chat_state)
    # text_input3 =  "the text is {} ".format(text) + text_input3
    # print(text_input3)
    chat_state, img_list, chatbot = upload_imgorvideo(video_path, image_path, text_input1, chat_state, chatbot, audio)
    _, chatbot, chat_state = gradio_ask(text_input1, chatbot, chat_state)
    chatbot, chat_state, img_list = gradio_answer(chatbot, chat_state, img_list, num_beams, temperature)
    print("\nchat_state is\n",chat_state)
    
    answers = [msg[1] for msg in chat_state.messages if msg[0] == 'ASSISTANT'][0]


    text_with_answer = text +' ' + ''.join(answers)

    result = {
            "Dialogue_ID": dialogue_id,
            "filename": filename,
            "speaker":speaker,
            "emotion":emotion,
            "strategy":strategy,
            "text":text,
            "answer": answers,
            "text_with_answer":text_with_answer

        }
    
    print(result)
    
    result_json_str = json.dumps(result, indent=4, ensure_ascii=False)



    with open(result_fpath, "w", encoding="utf-8") as f:
        f.write(result_json_str)

    chat_state, img_list = gradio_reset(chat_state, img_list)

    processed = processed+1
    # print(processed)

    print("processed {} rows".format(index + 1))


    # with open(json_fpath, "w", encoding="utf-8") as f:
    #     json_str = json.dumps(results, indent=4, ensure_ascii=False)
    #     f.write(json_str)

# with open(json_fpath, 'a', encoding="utf-8") as f:
#     f.write("\n]")

# print(f"Results have been saved to {json_fpath}.")

print("All Done!")