apt-get -y update
apt-get -y upgrade
apt-get -y install ffmpeg
python -m pip install --upgrade pip
pip uninstall Cython -y
pip install -r requirements.txt
cp /mnt/share/prosody_bert_model/ ./models/ -r
cp /mnt/share/tts_ecapa_exp/* ./ecapa/exps/ -r
cd vits_core/monotonic_align
python setup.py build_ext --inplace

