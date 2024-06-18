# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import json
import datetime
import torch
from torch import Tensor
import numpy as np
import os
import logging
import argparse
import random
import gradio as gr

from transformers.trainer_utils import set_seed
from utils.building_utils import boolean_string, build_model, deploy_model
from inputters import inputters
from inputters.inputter_utils import _norm

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset
from face_generate.face_demo import EAT


def cut_seq_to_eos(sentence, eos, remove_id=None):
    if remove_id is None:
        remove_id = [-1]
    sent = []
    for s in sentence:
        if s in remove_id:
            continue
        if s != eos:
            sent.append(s)
        else:
            break
    return sent


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True)
parser.add_argument('--inputter_name', type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_checkpoint", '-c', type=str, default=None)
parser.add_argument("--fp16", type=boolean_string, default=False)

parser.add_argument("--single_turn", action='store_true')
parser.add_argument("--max_input_length", type=int, default=150)
parser.add_argument("--max_src_turn", type=int, default=20)
parser.add_argument("--max_decoder_input_length", type=int, default=50)
parser.add_argument("--max_knl_len", type=int, default=64)
parser.add_argument('--label_num', type=int, default=None)

parser.add_argument("--min_length", type=int, default=5)
parser.add_argument("--max_length", type=int, default=50)

parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1)
parser.add_argument('--num_beams', type=int, default=1)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

parser.add_argument("--use_gpu", action='store_true')

# ========================================
#             Model Initialization
# ========================================

cmd = "--config_name lama --inputter_name lama --seed 3 --load_checkpoint ./DATA/lama.lama/2024-04-09205609.1e-05.16.1gpu-lama/epoch-9.bin --fp16 false --max_input_length 150 --max_decoder_input_length 50 --max_length 50 --min_length 10 --temperature 0.7 --top_k 0 --top_p 0.9 --num_beams 2 --repetition_penalty 1 --no_repeat_ngram_size 3"

import sys

sys.argv += cmd.split()

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
print("Using device:", device)
n_gpu = torch.cuda.device_count()
args.device, args.n_gpu = device, n_gpu

if args.load_checkpoint is not None:
    output_dir = args.load_checkpoint + '_interact_dialogs'
else:
    os.makedirs('./DEMO', exist_ok=True)
    output_dir = './DEMO/' + args.config_name
    if args.single_turn:
        output_dir = output_dir + '_1turn'
os.makedirs(output_dir, exist_ok=True)

#set_seed(args.seed)

names = {
    'inputter_name': args.inputter_name,
    'config_name': args.config_name,
}

toker, model, *_ = build_model(checkpoint=args.load_checkpoint, **names)
model = deploy_model(model, args)

model.eval()

inputter = inputters[args.inputter_name]()
dataloader_kwargs = {
    'max_src_turn': args.max_src_turn,
    'max_input_length': args.max_input_length,
    'max_decoder_input_length': args.max_decoder_input_length,
    'max_knl_len': args.max_knl_len,
    'label_num': args.label_num,
}


pad = toker.pad_token_id
if pad is None:
    pad = toker.eos_token_id
    assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
bos = toker.bos_token_id
if bos is None:
    bos = toker.cls_token_id
    assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
eos = toker.eos_token_id
if eos is None:
    eos = toker.sep_token_id
    assert eos is not None, 'either eos_token_id or sep_token_id should be provided'
    
generation_kwargs = {
    'max_length': args.max_length,
    'min_length': args.min_length,
    'do_sample': True if (args.top_k > 0 or args.top_p < 1) else False,
    'temperature': args.temperature,
    'top_k': args.top_k,
    'top_p': args.top_p,
    'num_beams': args.num_beams,
    'repetition_penalty': args.repetition_penalty,
    'no_repeat_ngram_size': args.no_repeat_ngram_size,
    'pad_token_id': pad,
    'bos_token_id': bos,
    'eos_token_id': eos,
}

eof_once = False
history = {'dialog': [],}

#audio
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model_speech = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state):
    if chat_state is not None:
        chat_state['dialog'] = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please choose your scnarios first', interactive=True),gr.update(value="Start Chat", interactive=True), chat_state

def upload_video(video_path, text_input, chat_state,chatbot):
    if video_path is None and text_input is None:
        return None, chat_state, None
    elif video_path is not None and text_input is not None:
        print()
        # video_html = "<video src='/file={}' style='width: 140px; max-width:none; max-height:none' preload='auto' controls></video>".format(video_path)
        chatbot = chatbot + [((video_path,), None)]
        return chat_state,chatbot

def generate_audio(text):
    # Process text input to generate speech
    inputs_audio = processor(text_target=text, return_tensors="pt")
    speech = model_speech.generate_speech(inputs_audio["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    # Save to a temporary file and return path (Gradio can handle files)
    file_path = "./face_generate/demo/video_processed/W015_neu_1_002/W015_neu_1_002.wav"
    sf.write(file_path, speech.numpy(), samplerate=16000)
    print("wav saved ...")

def generate_face(root_wav):
    name = "deepprompt_eam3d_all_final_313"
    eat = EAT(root_wav=root_wav)
    emo = "neu"
    face_output = './face_generate/demo/output/'
    eat.test(f'./face_generate/ckpt/{name}.pth.tar', emo, save_dir= face_output)
    mp4_files = []
    try:
        # Check if the directory exists and is accessible
        if os.path.exists(face_output) and os.path.isdir(face_output):
            # List all files in the directory
            for file in os.listdir(face_output):
                if file.endswith(".mp4"):
                    mp4_files.append(os.path.join(face_output, file))
    except Exception as e:
        print(f"An error occurred while listing MP4 files: {str(e)}")

    print(mp4_files)

    # Optionally return the list of MP4 file paths
    return mp4_files


def video_llama(gr_video, gr_img, text_input, chat_state,chatbot,audio_flag):
    if args.model_type == 'vicuna':
        chat_state = default_conversation.copy()
    else:
        chat_state = conv_llava_llama_2.copy()
    if gr_img is None and gr_video is None:
        return None, None, None, gr.update(interactive=True), chat_state, None
    elif gr_img is not None and gr_video is None:
        print(gr_img)
        chatbot = chatbot + [((gr_img,), None)]
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
        return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(value="Start Chatting", interactive=False), chat_state, img_list,chatbot
    else:
        # img_list = []
        return gr.update(interactive=False), gr.update(interactive=False),chat_state, None,chatbot


    
def gradio_ask(user_input, chatbot, chat_state):
    if not user_input.strip():
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    
    if chat_state is None:
        chat_state = {'dialog': []}

    # chatbot = chatbot + [[user_input, None]]


    normalized_input = _norm(user_input)
    chat_state['dialog'].append({
        'text': normalized_input,  # Assuming _norm is normalization function you might include
        'speaker': 'usr',
        'emotion': 'depression'
    })

    chat_state['dialog'].append({ # dummy tgt
        'text': 'n/a',
        'speaker': 'sys',
        'strategy':'Open question',
        'emotion':'neutral'
    })

    # Prepare the data for the model
    inputs = inputter.convert_data_to_inputs(chat_state, toker, **dataloader_kwargs)
    inputs = inputs[-1:]
    features = inputter.convert_inputs_to_features(inputs, toker, **dataloader_kwargs)
    batch = inputter.prepare_infer_batch(features, toker, interact=True)
    batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
    batch.update(generation_kwargs)

    # Remove batch_size from batch if exists
    batch.pop("batch_size", None)

    # Generate response from the model
    _, generations = model.generate(**batch)
    out = generations[0].tolist()
    out = cut_seq_to_eos(out, eos)
    response = toker.decode(out).encode('ascii', 'ignore').decode('ascii').strip()
    print(response)
    # chatbot[-1][1] = response

    chat_state['dialog'].pop()
    # Update history with the response
    chat_state['dialog'].append({
        'text': response,
        'speaker': 'sys',
        'strategy': 'Restatement',
        'emotion': 'neutral'
    })

    generate_audio(response)

    root_wav = "./face_generate/demo/video_processed/W015_neu_1_002"
    face_files = generate_face(root_wav)
    face_path = ("./face_generate/demo/output/Capture5_neu_W015_neu_1_002.mp4",)

    face_html = "<video src='/file={}' style='width: 140px; max-width:none; max-height:none' preload='auto' controls></video>".format(
       "./face_generate/demo/output/Capture5_neu_W015_neu_1_002.mp4"
    )


    chatbot = [
        *chatbot,
        (user_input, None),
        (None, face_html),
    ]


    # Clearing up the last dummy entry if used elsewhere
    return '', chatbot, chat_state


# title = """
# MimicTherapist: A Multimodal Emotional Support Chatbot for Therapeutic Counselling.
# MimicTherapist: A Multimodal Emotional Support Chatbot for better Emotion Interaction
# """

title = """
<style>
    .title {
        font-size: 32px; 
        font-weight: bold; 
        color: #333; 
        text-align: right; 
        margin: 10px 0;
    }
</style>
<div class='title'>Mi-Therapist: A Multimodal Emotional Support Counsellor</div>
"""

# scenarios_upload = ("""
# ### We provide some scenarios below, choose one of them similar to you. Simply click on them to try them out directly.
# """)

scenarios = [
    ("Breakups or Divorce", "examples/Breakup.mp4"),
    ("The relationship with friends and family", "examples/Relationship.mp4"),
    ("Dream Analysis", "examples/Dream_analysis.mp4"),
    ("Illness Coping","examples/Illness.mp4")
]

def update_text_input(scenario_content,video):
    if scenario_content == "Breakups or Divorce":
        text_input = "I left Andrew, I did some terrible things last night, I feel sick."
        input_video = video_llama(video, text_input)
        return input_video
    elif scenario_content == "The relationship with friends and family":
        text_input = "I've been under pressure for years, and she has never supported me. All the other mothers are right there, all over us visiting the gym, the training camps, all of it."
        input_video = video_llama(video, text_input)
        return input_video
    elif scenario_content == "Dream Analysis":
        text_input = "I had this very strange dream, and I was quite disturbed by it."
        input_video = video_llama(video, text_input)
        return input_video
    elif scenario_content == "Illness Coping":
        text_input = "But now I'm just lying in bed. I'm not sleeping. I'm not working."
        input_video = video_llama(video, text_input)
        return input_video
    elif scenario_content == "Childhood Shadow":
        text_input = "The move to Baltimore marked the end of my childhood. My father walked out; My mother got sicker, And I was forced to take care of her."
        input_video = video_llama(video, text_input)
        return input_video
    

# 'snehilsanyal/scikit-learn'
with gr.Blocks(theme='snehilsanyal/scikit-learn') as demo:
    gr.Markdown(title)
 
    with gr.Row():
        with gr.Column(scale=0.5):
            # scenarios = gr.Textbox(label='Scenarios')
            scenario_dropdown = gr.Dropdown(choices=[s[0] for s in scenarios], label="Scenarios")
            video_player = gr.Video()
            gr.Examples(examples=[[s[0], s[1]] for s in scenarios], inputs=[scenario_dropdown, video_player], label="Scenarios")

                                   
        with gr.Column():
            chat_state = gr.State()
            video_list = gr.State()
            chatbot = gr.Chatbot(label='Mi-Therapist',scale=5)
            text_input = gr.Textbox(label='User', placeholder='choose scenarios or directly start chat.', interactive=True)
            with gr.Row():
                upload_button = gr.Button(value="Start Chat", interactive=True, variant="primary")
                clear = gr.Button("Restart")


            scenario_dropdown.change(update_text_input, inputs=[scenario_dropdown, video_player], outputs=[text_input])

    # upload_button.click(upload_video, inputs=[video_player, text_input, chat_state, chatbot],
    #     outputs=[chat_state, chatbot])
    
    upload_button.click(gradio_ask, inputs=[text_input, chatbot, chat_state],
        outputs=[text_input,chatbot, chat_state])
    
    text_input.submit(gradio_ask, inputs=[text_input, chatbot, chat_state],
        outputs=[text_input,chatbot, chat_state])

    clear.click(
        gradio_reset,
        inputs=[chat_state],
        outputs= [chatbot, video_player, scenario_dropdown, text_input, upload_button, chat_state],
        queue=False
    )


demo.launch(share=False, enable_queue=True)


