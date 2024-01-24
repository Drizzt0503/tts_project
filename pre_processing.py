import os
import torch
import numpy as np
import argparse
import utils

from prosody_bert import TTSProsody
from prosody_bert.prosody_tool import is_chinese, pinyin_dict
from utils import load_wav_to_torch
from sound_processing.mel_processing import spectrogram_torch

lib_dir = os.path.dirname(os.path.realpath(__file__))
tdir=lib_dir+'/temp/train_set'
fdir=lib_dir+'/temp/filelists'
os.makedirs(tdir,exist_ok=True)
os.makedirs(fdir,exist_ok=True)
os.makedirs(tdir+"/waves", exist_ok=True)
os.makedirs(tdir+"/berts", exist_ok=True)
os.makedirs(tdir+"/temps", exist_ok=True)


def log(info: str):
    with open(ddir+'/prepare.log', "a", encoding='utf-8') as flog:
        print(info, file=flog)


def get_spec(hps, filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    assert sampling_rate == hps.data.sampling_rate, f"{sampling_rate} is not {hps.data.sampling_rate}"
    audio_norm = audio / hps.data.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = torch.squeeze(spec, 0)
    return spec


def bzn_to_temp():
    #check if BZNSYP exist?
    ddir=lib_dir+'/dataset/BZNSYP'
    #move .wav to train_set with ffmpeg
    wavs= os.listdir(ddir+'/waves')
    for wav in wavs:
        wpath=ddir+'/waves/'+wav
        w2path=tdir+'/waves/'+wav
        cmd =f'ffmpeg -y -i {wpath} -acodec pcm_s16le -ac 1 -ar 16000 {w2path}'
        os.system(cmd)
    to_text=[]
    with open(ddir+'/000001-010000.txt','rt') as ftext:
        while (True):
            try:
                message = ftext.readline().strip()
                pinyins = ftext.readline().strip()
            except Exception as e:
                print('nothing of except:', e)
                break
            if (message == None):
                break
            if (message == ""):
                break
            infosub = message.split("\t")
            fileidx = infosub[0]
            message = infosub[1]
            message = message.replace("#1", "")
            message = message.replace("#2", "")
            message = message.replace("#3", "")
            message = message.replace("#4", "")
            to_text.append(fileidx+'|'+message+'|'+pinyins)
    with open(tdir+'/text_info.txt','wt') as wtext:
        for any in to_text:
            wtext.write(any+'\n')


def pre_vits(config):
    hps = utils.get_hparams_from_file(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prosody = TTSProsody(lib_dir+"/models/prosody_bert_model", device)

    ftext = open(tdir+"/text_info.txt", "r+", encoding='utf-8')
    scrips = []
    while (True):
        try:
            text_info = ftext.readline().strip()
        except Exception as e:
            print('nothing of except:', e)
            break
        if (text_info == None):
            break
        if (text_info == ""):
            break
        text_info_t = text_info.split("|")

        fileidx = text_info_t[0]
        message = text_info_t[1]
        pinyins = text_info_t[2]

        try:
            phone_index = 0
            phone_items = []
            phone_items.append('sil')
            count_phone = []
            count_phone.append(1)

            pinyins = pinyins.split()
            len_pys = len(pinyins)
            for word in message:
                if is_chinese(word):
                    count_phone.append(2)
                    if (phone_index >= len_pys):
                        print(len_pys)
                        print(phone_index)
                    pinyin = pinyins[phone_index]
                    phone_index = phone_index + 1
                    if pinyin[:-1] in pinyin_dict:
                        tone = pinyin[-1]
                        a = pinyin[:-1]
                        a1, a2 = pinyin_dict[a]
                        phone_items += [a1, a2 + tone]
                else:
                    count_phone.append(1)
                    phone_items.append('sp')
            count_phone.append(1)
            phone_items.append('sil')
            phone_items_str = ' '.join(phone_items)
            #log(f"\t{phone_items_str}")
        except IndexError as e:
            print(f"{fileidx}\t{message}")
            print('except:', e)
            continue

        text = f'[PAD]{message}[PAD]'
        char_embeds = prosody.get_char_embeds(text)
        print(fileidx)
        char_embeds = prosody.expand_for_phone(char_embeds, count_phone)
        char_embeds_path = lib_dir+f"/temp/train_set/berts/{fileidx}.npy"
        np.save(char_embeds_path, char_embeds, allow_pickle=False)

        wave_path = lib_dir+f"/temp/train_set/waves/{fileidx}.wav"
        spec_path = lib_dir+f"/temp/train_set/temps/{fileidx}.spec.pt"
        spec = get_spec(hps, wave_path)
        torch.save(spec, spec_path)

        stemp=wave_path+'|'+spec_path+'|'+char_embeds_path+f'|{phone_items_str}'
        scrips.append(stemp)
    ftext.close()

    fout = open(lib_dir+f'/temp/filelists/all.txt', 'w', encoding='utf-8')
    for item in scrips:
        print(item, file=fout)
    fout.close()
    fout = open(lib_dir+f'/temp/filelists/valid.txt', 'w', encoding='utf-8')
    for item in scrips[:1]:
        print(item, file=fout)
    fout.close()
    fout = open(lib_dir+f'/temp/filelists/train.txt', 'w', encoding='utf-8')
    for item in scrips[:]:
        print(item, file=fout)
    fout.close()

def pre_vitsm(config):
    hps = utils.get_hparams_from_file(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prosody = TTSProsody(lib_dir+"/models/prosody_bert_model", device)

    ftext = open(tdir+"/text_info.txt", "r+", encoding='utf-8')
    scrips = []
    while (True):
        try:
            text_info = ftext.readline().strip()
        except Exception as e:
            print('nothing of except:', e)
            break
        if (text_info == None):
            break
        if (text_info == ""):
            break
        text_info_t = text_info.split("|")

        fileidx = text_info_t[0]
        message = text_info_t[1]
        pinyins = text_info_t[2]

        try:
            phone_index = 0
            phone_items = []
            phone_items.append('sil')
            count_phone = []
            count_phone.append(1)

            pinyins = pinyins.split()
            len_pys = len(pinyins)
            for word in message:
                if is_chinese(word):
                    count_phone.append(2)
                    if (phone_index >= len_pys):
                        print(len_pys)
                        print(phone_index)
                    pinyin = pinyins[phone_index]
                    phone_index = phone_index + 1
                    if pinyin[:-1] in pinyin_dict:
                        tone = pinyin[-1]
                        a = pinyin[:-1]
                        a1, a2 = pinyin_dict[a]
                        phone_items += [a1, a2 + tone]
                else:
                    count_phone.append(1)
                    phone_items.append('sp')
            count_phone.append(1)
            phone_items.append('sil')
            phone_items_str = ' '.join(phone_items)
            #log(f"\t{phone_items_str}")
        except IndexError as e:
            print(f"{fileidx}\t{message}")
            print('except:', e)
            continue

        text = f'[PAD]{message}[PAD]'
        char_embeds = prosody.get_char_embeds(text)
        print(fileidx)
        char_embeds = prosody.expand_for_phone(char_embeds, count_phone)
        char_embeds_path = lib_dir+f"/temp/train_set/berts/{fileidx}.npy"
        np.save(char_embeds_path, char_embeds, allow_pickle=False)

        wave_path = lib_dir+f"/temp/train_set/waves/{fileidx}.wav"
        spec_path = lib_dir+f"/temp/train_set/temps/{fileidx}.spec.pt"
        sp_emb_path = lib_dir+f"/temp/train_set/embed/{fileidx}.pt"
        spec = get_spec(hps, wave_path)
        torch.save(spec, spec_path)

        stemp=f"{wave_path}|{spec_path}|{char_embeds_path}|{phone_items_str}|{sp_emb_path}"
        scrips.append(stemp)
    ftext.close()

    fout = open(lib_dir+f'/temp/filelists/all.txt', 'w', encoding='utf-8')
    for item in scrips:
        print(item, file=fout)
    fout.close()
    fout = open(lib_dir+f'/temp/filelists/valid.txt', 'w', encoding='utf-8')
    for item in scrips[:1]:
        print(item, file=fout)
    fout.close()
    fout = open(lib_dir+f'/temp/filelists/train.txt', 'w', encoding='utf-8')
    for item in scrips[:]:
        print(item, file=fout)
    fout.close()

    os.makedirs(tdir+"/embed", exist_ok=True)
    cwd =os.getcwd()
    os.chdir(lib_dir+'/ecapa')
    os.system('python evalECAPA.py --register')
    os.chdir(cwd)


if __name__ == "__main__":
    #===here make train_set from default dataset
    #bzn_to_temp()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=lib_dir+"/config/bert_vits.json",
        help="JSON file for configuration",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="",
        help="If use some default dataset",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="vits",
        help="If use some default dataset",
    )
    args = parser.parse_args()
    if args.dataset == "BZNSYP":
        bzn_to_temp()
    else:
        pass
    if args.method =='vits':
        pre_vits(args.config)
    elif args.method =='vitsm':
        pre_vitsm(args.config)
    else:
        print('not supported method')

