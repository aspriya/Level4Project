import pyaudio
import struct
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
style.use("ggplot")

import matplotlib as mpl
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure


CHUNK = 2048 #Blocksize
CHANNELS = 1 #2
RATE = 44100  #Sampling Rate in Hz
RECORD_SECONDS = 70


# Create the figure we desire to add to an existing canvas
[fig, ax] = plt.subplots()
line, = ax.plot([], [], lw=1)
ax.set_ylim(-40000,40000)
ax.set_xlim(0, 2048 * 10)
plt.ylabel('Volume')
plt.xlabel('Samples')
plt.title('AUDIO WAVEFORM')


def init():
    line.set_data([], [])
    return line,

frames = [] #A python list of chunks (numpy.ndarray)
def animate(i):
    # update the data
    #Reading from audio input stream into data with block length "CHUNK":
    data = stream.read(CHUNK)
    frames.append(np.fromstring(data, dtype=np.int16))
    #Convert the list of numpy-arrays into a 1D array (column-wise)
    y = np.hstack(frames[-10:])
    x = np.linspace(0, len(y), len(y))
    line.set_data(x, y)
    return line,


p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

LARGE_FONT = ("Verdana", 12)

class AMT(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill = "both", expand = True)

        container.grid_rowconfigure(0, weight =1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        frame = StartPage(container, self)
        self.frames[StartPage] = frame
        frame.grid(row=0, column=0, sticky = "nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

def qf():
    print("you did it!")

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text = "Start Page", font = LARGE_FONT)
        label.pack (pady = 10, padx= 10)

        button1 = tk.Button(self, text="Visit Page 1", command = qf)
        button1.pack()


        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand= True)


        stream.stop_stream()
        stream.close()
        p.terminate()




app = AMT()
ani = animation.FuncAnimation(fig, animate,init_func= init, interval=1, blit=True)
print("ab")
app.mainloop()
