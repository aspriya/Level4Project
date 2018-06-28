import pyaudio
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
style.use("seaborn")

import wave
import sounddevice as sd

import matplotlib as mpl
import sys
from matplotlib.figure import Figure

from tkinter import * #if using python 3.* then use: from tkinter import * i.e (lowercase)
import tkinter.filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

import subprocess

import re

frames_nD = [] #A python list of chunks (numpy.ndarray)
frames = []

from PitchTranscription import *

CHUNK = 2048 #Blocksize
CHANNELS = 1 #2
RATE = 44100  #Sampling Rate in Hz
FORMAT = pyaudio.paInt16


# Create the live waver form figure we desire to add to an existing canvas
# [fig, ax] = plt.subplots()
fig = Figure(figsize=(4,3))
ax = fig.add_subplot(111)
line, = ax.plot([], [], lw=1)
ax.set_ylim(-40000,40000)
ax.set_xlim(0, 2048 * 10)
plt.ylabel('Volume')
plt.xlabel('Samples')
fig.suptitle('Live Audio Waveform')

animate_swith = True

def init():
    line.set_data([], [])
    return line,

def animate(i):
    if animate_swith:
        # update the data
        #Reading from audio input stream into data with block length "CHUNK":
        global frames_nD
        data = stream.read(CHUNK)
        frames_nD.append(np.fromstring(data, dtype=np.int16))
        #Convert the list of numpy-arrays into a 1D array (column-wise)
        y = np.hstack(frames_nD[-10:])
        x = np.linspace(0, len(y), len(y))
        line.set_data(x, y)
        return line,
    else:
        line.set_data([0,1], [0])
        return line,

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

LARGE_FONT = ("Verdana", 18)

class AMT(Frame):
    def __init__(self, master):

        Frame.__init__(self, master)
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        self.heading = Label(self, text="Automatic Music Transcriptor", font = LARGE_FONT)
        self.heading.grid(row=0, column=0, columnspan=5)

        self.canvas_live = FigureCanvasTkAgg(fig, self)
        self.canvas_live.show()
        self.canvas_live.get_tk_widget().grid(row = 1, column = 0, columnspan=5, rowspan=2)

        self.stop = Button(self, text="Stop Recording", command = self.stop_recording)
        self.stop.grid(row=3, column=0, sticky=W)

        self.stop = Button(self, text="Load", command = self.load_audio)
        self.stop.grid(row=3, column=1, sticky=W)



    def stop_recording(self):
        global animate_swith
        animate_swith = False
        stream.stop_stream()
        stream.close()
        p.terminate()

        global frames

        frames = np.hstack(frames_nD)
        frames = np.double(frames)
        frames = frames / (2 ** 15) #normalizing the sample to be in -1 to 1 range in value.
        DC = frames.mean()
        MAX = (np.abs(frames)).max()
        frames = (frames - DC) / (MAX + 0.0000000001)

        self.canvas_live.get_tk_widget().delete("all")
        print("frame 0 and 1 is: ", frames[1])

        # Create the wave form figure and add to canvas
        fig2 = Figure(figsize=(4,3))
        ax2 = fig2.add_subplot(111)
        line = ax2.plot(frames)
        plt.ylabel('Volume')
        plt.xlabel('Samples')
        fig2.suptitle('Full Audio Waveform')

        print(plt.style.available)

        self.canvas = FigureCanvasTkAgg(fig2, self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row = 1, column = 0, columnspan=5, rowspan=2)

        self.transliterate = Button(self, text="Play", command = lambda: play_audio(frames, RATE))
        self.transliterate.grid(row=3, column=2, sticky=W)

        self.transliterate = Button(self, text="Transliterate", command = lambda: transliterate(frames, RATE))
        self.transliterate.grid(row=3, column=3, sticky=W)

        self.transliterate = Button(self, text="Play Synthesized", command = play_synthesized)
        self.transliterate.grid(row=3, column=4, sticky=W)


    def load_audio(self):
        global animate_swith
        animate_swith = False
        global frames
        global RATE
        frames = []
        stream.stop_stream()
        stream.close()
        p.terminate()

        self.canvas_live.get_tk_widget().delete("all")

        audio_obj_path = tkinter.filedialog.askopenfilename(filetypes = (("wav files", ".*wav"), ("All files", "*.*")))

        audio_name = re.search(r"^.*/(.*\.wav)$", audio_obj_path).group(1)
        print("audio name: ", audio_name)

        spf = wave.open(audio_obj_path,'r')
        #Extract Raw Audio from Wav File

        RATE = spf.getframerate()
        print("sample rate of loaded auio is: ", RATE)

        frames = spf.readframes(-1)
        frames = np.fromstring(frames, 'Int16')

        frames = np.double(frames)
        frames = frames / (2 ** 15) #normalizing the sample to be in -1 to 1 range in value.
        DC = frames.mean()
        MAX = (np.abs(frames)).max()
        frames = (frames - DC) / (MAX + 0.0000000001)

        #If Stereo
        if spf.getnchannels() == 2:
            print('Just mono files')
            sys.exit(0)


        fig3 = Figure(figsize=(4,3))
        ax3 = fig3.add_subplot(111)
        line = ax3.plot(frames)
        plt.ylabel('Volume')
        plt.xlabel('Samples')
        fig3.suptitle(audio_name +' - Full Audio Waveform')

        self.canvas = FigureCanvasTkAgg(fig3, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row = 1, column = 0, columnspan=5, rowspan=2)

        self.transliterate = Button(self, text="Play", command = lambda: play_audio(frames, RATE))
        self.transliterate.grid(row=3, column=2, sticky=W)

        self.transliterate = Button(self, text="Transliterate", command = lambda: transliterate(frames, RATE))
        self.transliterate.grid(row=3, column=3, sticky=W)

        self.transliterate = Button(self, text="Play Synthesized", command = play_synthesized)
        self.transliterate.grid(row=3, column=4, sticky=W)


def transliterate(seframes, RATE):
    display_CQT(frames, RATE)
    detect_onsets(frames, RATE)
    get_synthesized_samples()
    plot_synthesized_CQT()
    g_tempo = estimate_global_tempo()
    audio_to_midi_melodia("/home/ashan/Desktop/test2_midi.mid", bpm=g_tempo)


    cmd2 = "midi2ly /home/ashan/Desktop/test2_midi.midi -o /home/ashan/Desktop/outfile.ly"
    resul2 = subprocess.call(cmd2, shell=True)
    # print("resul 2 : ", resul2)

    cmd4 = "cd /home/ashan/Desktop && lilypond --png outfile.ly"
    resul4 = subprocess.call(cmd4, shell = True)
    # print("resul4 : ", resul4)

    #remove last line entered by lilypond in the png by croping the png
    cmd5 = "cd /home/ashan/Desktop && convert outfile.png -gravity South -chop 0x45 outfile.png"
    resul5 = subprocess.call(cmd5, shell = True)
    plt.show()

root = Tk()
root.title("Welcom")

app = AMT(root)
ani = animation.FuncAnimation(fig, animate, init_func= init, interval=1, blit=True)

root.mainloop()
