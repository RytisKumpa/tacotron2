import sys
import os
sys.path.append('waveglow/')
import numpy as np
import torch
from scipy.io.wavfile import write

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
#from denoiser import Denoiser

hparams = create_hparams()
hparams.sampling_rate = 22050
sampling_rate = 22050
checkpoint_path = "checkpoint_21000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = 'waveglow_256channels.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
#denoiser = Denoiser(waveglow)

text = "Šį modelį tik treniravau dvi dienas ir net nenaudojau visų duomenų. Kai sukarpysiu vusas audio knygas ir pabaigsiu treniravimą, nebus įmanoma atskirti šio kompiuteriu sugeneruoto įrašo nuo žmogaus!"
sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()
	
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

with torch.no_grad():
    audio = 32768.0 * waveglow.infer(mel_outputs_postnet, sigma=0.666)[0]
	
	
audio = audio.cpu().numpy()
audio = audio.astype('int16')
audio_path = os.path.join('samples', "{}_synthesis.wav".format('sample'))
write(audio_path, sampling_rate, audio)
print(audio_path)
#audio_path = os.path.join('samples', "{}_synthesis.wav".format('sample'))
#save_audio_to_file(audio, 22050, audio_path)

#audio_denoised = denoiser(audio, strength=0.01)[:, 0]
#audio_path = os.path.join('samples', "{}_synthesis.wav".format('denoised_sample'))
#save_audio_to_file(audio, 22050, audio_path)