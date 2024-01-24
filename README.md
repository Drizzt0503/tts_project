# goal:
1. maintain easily 
2. git control
each dir corrspond to one service, and has his own git 
3. standard API
4. functional integration
5. error handle



# real goal
1. functional integration
2. standard API and documentation
2.a. error handle: input error, functional error, 
3. git control

# module 
vits chinese module
1. core: nececery file to construct model
model layer:attentions.py, modules.py, monotonic_align
function: transforms.py, commons.py
model: models.py
api: for infer, for train




1a.  side model: prosody_bert, ecapa


2. pre
api in: text, sound.
api out: to model.
text process: vits_pinyin.py, text
sound process: mel_processing.py
embedding process:
data_utils.py, vits_prepare.py

flow:
python pre_vits.py --dataset BZN --model vits --config config
if None user has to build dataset in stardard way
1. from dataset to temp
2. prepare in temp:text spec wave bert embedding
3. into filelists




3. train
losses.py, utils.py, train.py
4. infer
vits_infer.py


### install
write a shell script in the main dir to install for different environment

conda create -n openvoice python=3.9
conda activate openvoice
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt

cd monotonic_align

python setup.py build_ext --inplace


### next to do
remove bert module from Vits

input licence

handle text

vist & vitsm duration predictor same but different code.
need to generate new default vitsm model
now use temp_model

vits_pinyin need to separate into pinyin and bert

# temp
outter example audio:
/var/www/html/webuploader/audio_temp

in db example audio:
/home/yuhang/tts_dataset

concate model and speaker_embedding into one file
