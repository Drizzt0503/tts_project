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
from text.phoneme_table import tailao_dic, _pause_t

lib_dir = os.path.dirname(os.path.realpath(__file__))
tdir=lib_dir+'/temp/train_set'
fdir=lib_dir+'/temp/filelists'
os.makedirs(tdir,exist_ok=True)
os.makedirs(fdir,exist_ok=True)
os.makedirs(tdir+"/waves", exist_ok=True)
os.makedirs(tdir+"/temps", exist_ok=True)


def get_phon(text3):
    to_save=0
    plist=text3.strip().split(' ')
    for che in plist:
        if che in _pause_t:
            pass
        else:
            if che[:-1] in tailao_dic:
                pass
            else:
                to_save=1
                print(text3)
                print(che)
                print(plist)
                print('==============')
    if to_save == 0:
        p2=text3.strip().split()
        p3=[]
        for pho in p2:
            if pho in _pause_t:
                p3.append(pho+'_t')
            elif pho[:-1] in tailao_dic:
                a1,b1 = tailao_dic[pho[:-1]]
                p3.append(a1+'_t')
                p3.append(b1+pho[-1]+'_t')
            else:
                print('not in dic')
        p_str='sil_t '
        for ppho in p3:
            p_str+= ppho+ ' '
        p_str += 'sil_t'
        return p_str
    else:
        return False




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



def pre_suisiann(config):
    hps = utils.get_hparams_from_file(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ftext = open(tdir+"/suisiann_text_r.txt", "r+", encoding='utf-8')
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
        textn = text_info_t[2]
        abc = textn.replace('─',' split ')\
            .replace('……',' split split')\
            .replace('…',' split ')\
            .replace(',',' split ')\
            .replace('?',' split ')\
            .replace('!',' split ')\
            .replace(';',' split ')\
            .replace(':',' split ')\
            .replace('.',' split ')\
            .replace('-',' conn ')
        abc = abc.replace('   ',' ')\
            .replace('  ',' ')
        #print(abc)
        bbc =get_phon(abc)
        #print('======================')
        #print(bbc)
        #print('======================')
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
        sp_emb_path = lib_dir+f"/temp/train_set/embed/{fileidx}.pt"
        spec = get_spec(hps, wave_path)
        torch.save(spec, spec_path)
        plist=bbc.split(' ')
        lan_emb = [2]*len(plist)
        lan_str=''
        for b in lan_emb:
            lan_str+=str(b)+' '
        lan_str=lan_str.strip()
        stemp=f"{wave_path}|{spec_path}|{bbc}|{sp_emb_path}|{lan_str}|2"

        #stemp=f"{wave_path}|{spec_path}|{bbc}|{sp_emb_path}"
        scrips.append(stemp)
        #print(nn)
        nn+=1
    ftext.close()

    fout = open(lib_dir+f'/temp/filelists/all_t.txt', 'w', encoding='utf-8')
    for item in scrips:
        print(item, file=fout)
    fout.close()
    fout = open(lib_dir+f'/temp/filelists/valid_t.txt', 'w', encoding='utf-8')
    for item in scrips[:1]:
        print(item, file=fout)
    fout.close()
    fout = open(lib_dir+f'/temp/filelists/train_t.txt', 'w', encoding='utf-8')
    for item in scrips[:]:
        print(item, file=fout)
    fout.close()
"""
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
        #char_embeds = prosody.get_char_embeds(text)
        print(fileidx)
        #char_embeds = prosody.expand_for_phone(char_embeds, count_phone)
        #char_embeds_path = lib_dir+f"/temp/train_set/berts/{fileidx}.npy"
        #np.save(char_embeds_path, char_embeds, allow_pickle=False)

        wave_path = lib_dir+f"/temp/train_set/waves/{fileidx}.wav"
        spec_path = lib_dir+f"/temp/train_set/temps/{fileidx}.spec.pt"
        spec = get_spec(hps, wave_path)
        torch.save(spec, spec_path)

        stemp=wave_path+'|'+spec_path+f'|{phone_items_str}'
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
"""


if __name__ == "__main__":
    #===here make train_set from default dataset
    #bzn_to_temp()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=lib_dir+"/config/vits_multi.json",
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
        default="suisiann",
        help="If use some default dataset",
    )
    args = parser.parse_args()
    if args.dataset == "BZNSYP":
        bzn_to_temp()
    else:
        pass
    if args.method =='suisiann':
        pre_suisiann(args.config)
