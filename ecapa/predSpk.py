import os
import argparse, glob, torch, warnings, time
from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=500,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=12,      help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')
## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/data2/Evan/ECAPA/voxCeleb/list/train_list.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str,   default="/data2/Evan/ECAPA/vox2/",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--eval_list',  type=str,   default="/data2/Evan/ECAPA/voxCeleb/list/list_test_all2.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="/data2/Evan/ECAPA/vox1/",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--musan_path', type=str,   default="/data2/Evan/ECAPA/musanSplit/",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="/data2/Evan/ECAPA/rirsNoises/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser.add_argument('--save_path',  type=str,   default="/root/ECAPA-TDNN/exps/exp3",                                     help='Path to save the score.txt and models')
## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')

## Command
parser.add_argument('--eval'    , default=False , dest='eval', action='store_true', help='Only do evaluation')
parser.add_argument('--register', default=False , dest='register', action='store_true', help='Register speaker')
parser.add_argument('--initial_model',  type=str,   default="/root/ECAPA-TDNN/exps/exp/model/model_0060.model",                                          help='Path of the initial_model')

## eval path
parser.add_argument('--audio_path',  type=str,   default="/data2/Evan/ECAPA/vox1/", help='your audio path')
parser.add_argument('--spkid_path',  type=str,   default="./register/", help='speaker ID storage path')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

args.register = False
args.eval = True
s = ECAPAModel(**vars(args))
print("Model %s loaded from previous state!"%args.initial_model)
s.load_parameters(args.initial_model)

if args.register == True :
	spkID_lst = ['小聶','王晴蒂','Ken']
	audio_lst = ['onlyAudio_split1.wav', 'onlyAudio_split2.wav', 'onlyAudio_split3.wav']
	# audio path = audioPath + audio_lst[0 ~ n], where n is len(audio_lst)
	s.spk_register(spkID_lst, audio_lst, spkidPath = args.spkid_path, audioPath = './sound_meet/')




