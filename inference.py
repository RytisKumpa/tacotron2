import sys
#sys.path.append('waveglow/')
import numpy as np
import torch
import os

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
#from stft import STFT
from audio_processing import griffin_lim
from audio_utilities import reconstruct_signal_griffin_lim, save_audio_to_file
from train import load_model
from text import text_to_sequence
#from denoiser import Denoiser
from scipy.io.wavfile import write

hparams = create_hparams()
hparams.fp16_run = True
hparams.filter_length = 1024
hparams.hop_length = 256
win_length = 1024
hparams.sampling_rate = 22050

checkpoint_path = "checkpoint_21000"
#checkpoint_path = "tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval()#.half()

#waveglow_path = 'waveglow_256channels.pt'
#waveglow = torch.load(waveglow_path)['model']
#waveglow.cuda().eval().half()
#for k in waveglow.convinv:
#    k.float()
#denoiser = Denoiser(waveglow)

#text = "Testuojame modelį! Ar jis tikrai veikia? Ne."
text = "Testing, testing, testing."
sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

#with torch.no_grad():
#    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
#mel = mel_outputs.unsqueeze(0)
#mel = torch.autograd.Variable(mel[0].cuda())
mel = mel_outputs.float().data.cpu().numpy()[0]
#mel = mel_outputs.detach().cpu().numpy()[0]
#mel = mel_outputs[0].cuda()
#mel = torch.unsqueeze(mel, 0)
#mel = mel.half()
#stft_fn = STFT(filter_length=158, hop_length=258, win_length=158)
#stft_fn = stft_fn.half()
#stft_fn_cuda = stft_fn.cuda()
#audio = griffin_lim(mel, stft_fn, 300)#.detach()

audio = reconstruct_signal_griffin_lim(mel.T, mel.shape[0]*2-2, 256, 300)
#audio = audio / audio.max()

#audio = denoiser(audio, strength=0.01)[:, 0]
#audio = audio.squeeze()
#audio = audio[0]

#audio = audio.cpu().numpy()
audio = audio / audio.max()
audio = audio.astype('int16')
audio_path = os.path.join('samples', "{}_synthesis.wav".format('sample'))
save_audio_to_file(audio, 22050, audio_path)
#write(audio_path, hparams.sampling_rate, audio)
print(audio_path)