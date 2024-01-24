import os
def videoPre():
	inp_file = 'Untitled_Project2.mp4'
	opt_file = 'onlyAudio2.aac'
	wav_file = 'onlyAudio2.wav'
	res_file = 'onlyAudio2_re.wav'
	os.system('ffmpeg -y -i ./'+inp_file+' -acodec copy -vn ./'+opt_file)

	os.system('ffmpeg -y -i ./'+opt_file+' ./'+wav_file)

	os.system('ffmpeg -y -i ./'+wav_file+' -ar 16000 ./'+res_file)
def soundSplit():
	inp_file = 'onlyAudio2_re.wav'
	opt_file = 'onlyAudio_split3.wav'
	os.system('ffmpeg -i '+inp_file+' -ss 00:01:20 -to 00:01:24 '+opt_file)


soundSplit()
