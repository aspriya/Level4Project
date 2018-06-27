
import matplotlib.pyplot as plt

import numpy as np


#enamble caching of librosa for speed up processes
# allows you to store and re-use intermediate computations across sessions
import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
import librosa, librosa.display

import argparse


from midiutil.MidiFile import MIDIFile


import sounddevice as sd

# from progressbar import ProgressBar

# pbar = ProgressBar()



# #### Play the audio file.
def play_audio(frames, RATE):
    global x
    global sr
    x = frames
    sr = RATE
    sd.play(frames, RATE)


# #### Display the CQT (Constant Q Transform) of the signal.
def display_CQT(frames, RATE):
    global x
    global sr
    x = frames
    sr = RATE
    bins_per_octave = 36
    cqt = librosa.cqt(x, sr=sr, n_bins=300, bins_per_octave=bins_per_octave)
    log_cqt = librosa.amplitude_to_db(librosa.magphase(cqt)[0])

    print("CQT shape is", cqt.shape)
    ax1 = plt.subplot(4,1,1)
    librosa.display.specshow(log_cqt, sr=sr, x_axis='time', y_axis='cqt_note', bins_per_octave=bins_per_octave)



# #### Goal: Identify the pitch of each note and replace each note with a pure tone of that pitch.

# ## Step 1: Detect Onsets
# #### To accurately detect onsets, it may be helpful to see what the novelty funYction looks like:
onset_boundaries = []
onset_times = []
def detect_onsets(x, sr):
    print("\n\ntype of data is :",type(x))
    global hop_length # hop lenght = samples per frame
    global onset_envelope
    hop_length = 100
    onset_envelope = librosa.onset.onset_strength(x, sr = sr, hop_length=hop_length, n_fft= 2048)

    ax2 = plt.subplot(4,1, 2)
    plt.plot(onset_envelope)
    plt.xlim(0, len(onset_envelope))

    global onset_boundaries
    global onset_times

    # #### Among the obvious large peaks, there are many smaller peaks. We want to choose parameters which preserve the large peaks while ignoring the small peaks.  Next, we try to detect onsets.
    onset_samples = librosa.onset.onset_detect(x,
                                               sr=sr,
                                               onset_envelope= onset_envelope,
                                               units='samples',
                                               hop_length=hop_length,
                                               backtrack=True,
                                               pre_max=5,
                                               post_max=6,
                                               pre_avg=100,
                                               post_avg=100,
                                               delta=0.2,
                                               wait=0)

    print("Onset Samples: ", onset_samples)


    # #### Let's pad the onsets with the beginning and end of the signal.
    onset_boundaries = np.concatenate([[0], onset_samples, [len(x)]])
    print("onset boundaries in sample numbers: ",onset_boundaries)


    # #### Convert the onsets to units of seconds:
    onset_times = librosa.samples_to_time(onset_boundaries, sr=sr)
    print("onset times: ", onset_times)
    print("number of detected onset samples without start and end :", len(onset_samples))

    # #### Display the results of the onset detection:
    ax3 = plt.subplot(4,1, 3)
    librosa.display.waveplot(x, sr=sr)
    plt.vlines(onset_times, -1, 1, color='r')



# ## Step 2: Estimate Pitch
# #### Estimate pitch using the autocorrelation method:
f0s = [] #f0s array
def estimate_pitch(n0, n1, fmin=50.0, fmax=2000.0): #F0 ESTIMATION OF A GIVEN SEGMENT

    # Compute autocorrelation of input segment.
    segment = x[n0:n1]
    r = librosa.autocorrelate(segment)

    # Define lower and upper limits for the autocorrelation argmax.
    i_min = sr/fmax
    i_max = sr/fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0

    # Find the location of the maximum autocorrelation.
    i = r.argmax()
    f0 = float(sr)/i
    f0s.append(f0)
    return f0


# ## Step 3: Generate Pure Tone
# #### Create a function to generate a pure tone at the specified frequency:
def estimate_pitch_and_generate_sine(i):
    n0 = onset_boundaries[i]
    n1 = onset_boundaries[i+1]
    f0 = estimate_pitch(n0,n1)
    n = np.arange(n1-n0)
    return 0.2*np.sin(2*np.pi*f0*n/float(sr))


# #### Use a list comprehension to concatenate the synthesized segments:
def get_synthesized_samples():
    print("started synthesizing and estimating f0 using autocorrelation..")
    global y
    y = np.concatenate([
        estimate_pitch_and_generate_sine(i)
        for i in range(len(onset_boundaries)-1) #for i in pbar(range(len(onset_boundaries)-1))
    ])

    print(("synthesized!"))


# #### Play the synthesized transcription.
def play_synthesized():
    if(len(y) > 0):
        sd.play(y, sr)


# #### Plot the CQT of the synthesized transcription.
def plot_synthesized_CQT():
    global syn_cqt
    syn_cqt = librosa.cqt(y, sr=sr)
    ax4 = plt.subplot(4, 1, 4)
    librosa.display.specshow(abs(syn_cqt), sr=sr, x_axis='time', y_axis='cqt_note')



def estimate_global_tempo(): #tempo is the bpm (beats per minuit)
    tempo = librosa.beat.tempo(x, sr=sr, onset_envelope=onset_envelope)
    print("tempo is : ", tempo)
    return tempo




# #### Now convert the synthesized y audio to midi with the help of "audio to melodia file". then convert that midi to sheet music.

# #### write synthesized signal

# pip install MIDIUtil

def save_midi(outfile, notes, tempo):

    track = 0
    time = 0
    midifile = MIDIFile(1)

    # Add track name and tempo.
    midifile.addTrackName(track, time, "")
    midifile.addTempo(track, time, tempo)

    channel = 0
    volume = 100

    for note in notes:
        onset = note[0] * (tempo/60.)
        duration = note[1] * (tempo/60.)
        # duration = 1
        pitch = int(note[2].item())
        midifile.addNote(track, channel, pitch, onset, duration, volume)

    # And write it to disk.
    binfile = open(outfile, 'wb')
    midifile.writeFile(binfile)
    binfile.close()


def hz2midi(hz):

    # convert from Hz to midi note
    hz_nonneg = hz.copy()
    idx = hz_nonneg <= 0
    hz_nonneg[idx] = 1
    midi = 69 + 12*np.log2(hz_nonneg/440.)
    midi[idx] = 0

    # round
    midi = np.round(midi)

    return midi


def generate_note_sequance(midi_notes):  #one of my algorithm
    note_seqance = []
    for i, v in enumerate(midi_notes):
        note_seqance.append([onset_times[i], onset_times[i+1] - onset_times[i], v])

    return note_seqance



def audio_to_midi_melodia(outfile, bpm, smooth=0.15, minduration=0.1):

    # convert f0 to midi notes
    print("Converting Hz to MIDI notes...")
    midi_pitch = hz2midi(np.asarray(f0s))

    # segment sequence into individual midi notes
    notes = generate_note_sequance(midi_pitch)


    # save note sequence to a midi file
    print("Saving MIDI to disk...")
    save_midi(outfile, notes, bpm)

    print("Conversion complete.")
