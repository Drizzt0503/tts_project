import os
import torch
import numpy as np
import argparse
import utils

from prosody_bert import TTSProsody
from prosody_bert.prosody_tool import is_chinese, pinyin_dict
from utils import load_wav_to_torch
from sound_processing.mel_processing import spectrogram_torch
import text
from text.phoneme_table import _pad, _punctuation, _letters, _letters_ipa

lib_dir = os.path.dirname(os.path.realpath(__file__))
tdir=lib_dir+'/temp/train_set'
fdir=lib_dir+'/temp/filelists'
os.makedirs(tdir,exist_ok=True)
os.makedirs(fdir,exist_ok=True)
os.makedirs(tdir+"/waves", exist_ok=True)
os.makedirs(tdir+"/temps", exist_ok=True)




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



def pre_ljs(config):
    hps = utils.get_hparams_from_file(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ftext = open(tdir+"/text_info.txt", "r+", encoding='utf-8')
    scrips = []
    nn=1
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
        text0 = text_info_t[1]
        textn = text_info_t[2]
        temp1 = text._clean_text(text0, ["english_cleaners2"])
        #temp1 = text._clean_text(text0, ["english_cleaners2"])
        #temp3 = text._clean_text(text0, ["english_cleaners"])
        #temp4 = text._clean_text(textn, ["english_cleaners"])
        #print(fileidx)
        #print(text0)
        #print(temp1)
        #print(temp3)
        #print('========================')
        wave_path = lib_dir+f"/temp/train_set/waves/{fileidx}.wav"
        spec_path = lib_dir+f"/temp/train_set/temps/{fileidx}.spec.pt"
        spec = get_spec(hps, wave_path)
        torch.save(spec, spec_path)

        stemp=wave_path+'|'+spec_path+f'|{temp1}'
        scrips.append(stemp)
        print(nn)
        nn+=1
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

def pre_vctk(config):
    hps = utils.get_hparams_from_file(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ftext = open(tdir+"/vctk_text_r.txt", "r+", encoding='utf-8')
    scrips = []
    nn=1
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
        fileidx = text_info_t[0][:-4]
        text0 = text_info_t[1]
        temp1 = text._clean_text(text0, ["english_cleaners2"])
        #temp1 = text._clean_text(text0, ["english_cleaners2"])
        #temp3 = text._clean_text(text0, ["english_cleaners"])
        #temp4 = text._clean_text(textn, ["english_cleaners"])
        #print(fileidx)
        #print(text0)
        #print(temp1)
        #print(temp3)
        #print('========================')
        punc = _punctuation[:-1]
        abc = temp1.split()
        #print(abc)
        #print('=========')
        p_str='sil_e '
        for word in abc:
            for sym in word:
                p_str+=sym+' '
            p_str+='sp_e '
        p_str+= 'sil_e'
        wave_path = lib_dir+f"/temp/train_set/waves/{fileidx}.wav"
        spec_path = lib_dir+f"/temp/train_set/temps/{fileidx}.spec.pt"
        sp_emb_path = lib_dir+f"/temp/train_set/embed/{fileidx}.pt"
        spec = get_spec(hps, wave_path)
        torch.save(spec, spec_path)

        stemp=f"{wave_path}|{spec_path}|{p_str}|{sp_emb_path}"
        #stemp=wave_path+'|'+spec_path+f'|{temp1}'
        scrips.append(stemp)
        print(nn)
        nn+=1
    ftext.close()

    fout = open(lib_dir+f'/temp/filelists/all_e.txt', 'w', encoding='utf-8')
    for item in scrips:
        print(item, file=fout)
    fout.close()
    fout = open(lib_dir+f'/temp/filelists/valid_e.txt', 'w', encoding='utf-8')
    for item in scrips[:1]:
        print(item, file=fout)
    fout.close()
    fout = open(lib_dir+f'/temp/filelists/train_e.txt', 'w', encoding='utf-8')
    for item in scrips[:]:
        print(item, file=fout)
    fout.close()

    #os.makedirs(tdir+"/embed", exist_ok=True)
    #cwd =os.getcwd()
    #os.chdir(lib_dir+'/ecapa')
    #os.system('python evalECAPA.py --register')
    #os.chdir(cwd)

if __name__ == "__main__":
    #===here make train_set from default dataset
    #bzn_to_temp()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=lib_dir+"/config/vits_e.json",
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
        default="vctk",
        help="If use some default dataset",
    )
    args = parser.parse_args()
    if args.dataset == "BZNSYP":
        bzn_to_temp()
    else:
        pass
    if args.method =='ljs':
        pre_ljs(args.config)
    if args.method =='vctk':
        pre_vctk(args.config)
