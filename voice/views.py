from django.contrib import messages
from django.http.response import JsonResponse
from django.shortcuts import get_object_or_404, render,redirect
import os
from django.conf import settings
import scipy.io.wavfile as wavfile
from scipy.io.wavfile import write
import shutil
from .models import Record,Generated
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import matplotlib.pyplot as plt
import subprocess
import librosa
import time
from pydub import AudioSegment
from encoder import inference as encoder
from synthesizer import inference as Synthesizer
from vocoder import inference as vocoder

def record(request):
    if request.method == "POST":
        audio_file = request.FILES.get("recorded_audio")
        print(audio_file)
        record = Record.objects.create(
            voice_record=audio_file)
        record.save()
        messages.success(request, "Audio recording successfully added!")
        return JsonResponse(
            {
                "url": record.get_absolute_url(),
                "success": True,
            }
        )
    context = {"page_title": "Record audio"}
    return render(request, "core/record.html", context)


def record_detail(request, id):
    record = get_object_or_404(Record, id=id)
    context = {
        "page_title": "Recorded audio detail",
        "recordd": record,
        "id":id,
    }
    return render(request, "core/record_detail.html", context)


class AudioGenerator:
    def __init__(self):
        self.model_fpath = Path("trained_models/synthesizer.pt")
        self.vocoder_model_fpath = Path("trained_models/vocoder.pt")
        vocoder.load_model(self.vocoder_model_fpath)
    def make_embedding(self, wav):
        # make mel-spectrograrm
        # spec = Synthesizer.make_spectrogram(wav)
        # self.ui.draw_spec(spec, "current")

        # calculate embeddingg
        model_fpath = ('trained_models/encoder.pt')
        encoder.load_model(model_fpath)
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(
            encoder_wav, return_partials=True)
        return embed
    def process_text(self,text,embedd):
        text_specs = Synthesizer.Synthesizer(self.model_fpath)
        specs = text_specs.synthesize_spectrograms(text,embedd)
        return specs

    def vocode(self, spec, breaks):
        wav = vocoder.infer_waveform(
            spec, progress_callback=None)
        b_ends = np.cumsum(np.array(breaks) * 200)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * 16000))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])
        wav = encoder.preprocess_wav(wav)
        print("DONE")
        print(wav.shape)
        sd.play(wav, 16000)
        sd.wait()

        final_audio = "final_audio" + \
            "_rec_%05d" % np.random.randint(100000)+".wav"
        sf.write(final_audio,wav,16000)
        print(final_audio)
        id =np.random.randint(10000000)
        generatedaudio = Generated.objects.create(
            generated_voice=final_audio,id=id)
        generatedaudio.save()
        print(id)
        media_root = settings.MEDIA_ROOT
        media_path = os.path.join(media_root, 'generated')
        media_file = os.path.join(media_path, final_audio)

        # Create the media directory if it doesn't exist
        if not os.path.exists(media_path):
            os.makedirs(media_path)

        # Move the file to the media directory
        shutil.move(final_audio, media_file)
        print(media_file)
        context={
            "generated_file":media_file
        }
        
        return context
        

    def __call__(self, request):
        if request.method == 'POST':
            id = request.POST.get('id')
            text = request.POST.get('speechtext')
            aud = Record.objects.get(id=id)
            audio_file = aud.voice_record
            aud_path = audio_file.path
            # audio_path=os.path.basename(aud_path)
            input_audio = "input_audio"+ "_rec_%05d" % np.random.randint(100000)+".wav"
            subprocess.run(['ffmpeg', '-i', aud_path, '-vn', '-acodec',
                           'pcm_s16le', '-ar', '16000', '-ac', '1', input_audio])
            audiorecord, sample_rate = sf.read(input_audio)
            print(sample_rate)
            print(type(audiorecord))
            print(audiorecord.shape)
            audiorecord=audiorecord[:-640]
            print(audiorecord.shape)
            sd.play(audiorecord, sample_rate)
            sd.wait()
            print(text)
            speaker_embedding = self.make_embedding(audiorecord)
            print(speaker_embedding)
            texts=[text,'']
            embeddings = [speaker_embedding]*len(texts)
            spectrogram=self.process_text(texts,embeddings)
            spec = np.concatenate(spectrogram, axis=1)
            breaks = [spec.shape[1] for spec in spectrogram]
            context=self.vocode(spec,breaks)


        else:
            return redirect('')
        
        return render(request, "core/final.html",context)

generate_audio = AudioGenerator()
