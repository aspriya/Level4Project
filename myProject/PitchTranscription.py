from scipy import signal
from scipy.io import wavfile #to read and write wavfiles
import matplotlib.pyplot as plt
import seaborn
import numpy, scipy
import librosa, librosa.display

import vamp
import argparse
import os
import numpy as np
from midiutil.MidiFile import MIDIFile
from scipy.signal import medfilt
import jams

import sounddevice as sd



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
def detect_onsets(x, sr):
    print("\n\ntype of data is :",type(x))
    global hop_length # hop lenght = samples per frame
    hop_length = 150
    onset_envelope = librosa.onset.onset_strength(x, sr = sr, hop_length=hop_length, n_fft= 1024)

    ax2 = plt.subplot(4,1, 2)
    plt.plot(onset_envelope)
    plt.xlim(0, len(onset_envelope))

    global onset_boundaries

    # #### Among the obvious large peaks, there are many smaller peaks. We want to choose parameters which preserve the large peaks while ignoring the small peaks.  Next, we try to detect onsets.
    onset_samples = librosa.onset.onset_detect(x,
                                               sr=sr,
                                               onset_envelope= onset_envelope,
                                               units='samples',
                                               hop_length=hop_length,
                                               backtrack=True,
                                               pre_max=5,
                                               post_max=6,
                                               pre_avg=60,
                                               post_avg=60,
                                               delta=0.2,
                                               wait=0)

    print("Onset Samples: ", onset_samples)


    # #### Let's pad the onsets with the beginning and end of the signal.
    onset_boundaries = numpy.concatenate([[0], onset_samples, [len(x)]])
    print("onset boundaries in sample numbers: ",onset_boundaries)


    # #### Convert the onsets to units of seconds:
    onset_times = librosa.samples_to_time(onset_boundaries, sr=sr)
    print("onset times: ", onset_times)


    # #### Display the results of the onset detection:
    ax3 = plt.subplot(4,1, 3)
    librosa.display.waveplot(x, sr=sr)
    plt.vlines(onset_times, -1, 1, color='r')



# ## Step 2: Estimate Pitch
# #### Estimate pitch using the autocorrelation method:
f0s = []
def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0): #F0 ESTIMATION OF A GIVEN SEGMENT

    # Compute autocorrelation of input segment.
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
def generate_sine(f0, sr, n_duration):
    n = numpy.arange(n_duration)
    return 0.2*numpy.sin(2*numpy.pi*f0*n/float(sr))


# ## Step 4: Synthesize and plot synthesized CQT
# #### Create a helper function for use in a list comprehension:
def estimate_pitch_and_generate_sine(x, onset_samples, i, sr):
    n0 = onset_samples[i]
    n1 = onset_samples[i+1]
    f0 = estimate_pitch(x[n0:n1], sr)
    return generate_sine(f0, sr, n1-n0)


# #### Use a list comprehension to concatenate the synthesized segments:
def get_synthesized_samples():
    global y
    y = numpy.concatenate([
        estimate_pitch_and_generate_sine(x, onset_boundaries, i, sr=sr)
        for i in range(len(onset_boundaries)-1)
    ])
    print("synsthesized first 20 samples: ", y[0:20])


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
    plt.show()


def estimate_global_tempo(): #tempo is the bpm (beats per minuit)
    tempo = librosa.beat.tempo(x, sr=sr)
    print("tempo is : ", tempo)
    return tempo




# #### Now convert the synthesized y audio to midi with the help of "audio to melodia file". then convert that midi to sheet music.

# #### write synthesized signal

# pip install vamp
# pip install MIDIUtil
# pip install --user jams

def save_jams(jamsfile, notes, track_duration, orig_filename):

    # Construct a new JAMS object and annotation records
    jam = jams.JAMS()

    # Store the track duration
    jam.file_metadata.duration = track_duration
    jam.file_metadata.title = orig_filename

    midi_an = jams.Annotation(namespace='pitch_midi',
                              duration=track_duration)
    midi_an.annotation_metadata =         jams.AnnotationMetadata(
            data_source='audio_to_midi_melodia.py v%s' % __init__.__version__,
            annotation_tools='audio_to_midi_melodia.py (https://github.com/'
                             'justinsalamon/audio_to_midi_melodia)')

    # Add midi notes to the annotation record.
    for n in notes:
        midi_an.append(time=n[0], duration=n[1], value=n[2], confidence=0)

    # Store the new annotation in the jam
    jam.annotations.append(midi_an)

    # Save to disk
    jam.save(jamsfile)


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
        pitch = note[2]
        midifile.addNote(track, channel, pitch, onset, duration, volume)

    # And write it to disk.
    binfile = open(outfile, 'wb')
    midifile.writeFile(binfile)
    binfile.close()


def midi_to_notes(midi, fs, hop, smooth, minduration):

    # smooth midi pitch sequence first
    if (smooth > 0):
        filter_duration = smooth  # in seconds
        filter_size = int(filter_duration * fs / float(hop))
        if filter_size % 2 == 0:
            filter_size += 1
        midi_filt = medfilt(midi, filter_size) #applying a median filter
    else:
        midi_filt = midi
    # print(len(midi),len(midi_filt))

    notes = []
    p_prev = None
    duration = 0
    onset = 0
    for n, p in enumerate(midi_filt):
        if p == p_prev:
            duration += 1
        else:
            # treat 0 as silence
            if p_prev > 0:
                # add note
                duration_sec = duration * hop / float(fs)
                # only add notes that are long enough
                if duration_sec >= minduration:
                    onset_sec = onset * hop / float(fs)
                    notes.append((onset_sec, duration_sec, p_prev))

            # start new note
            onset = n
            duration = 1
            p_prev = p

    # add last note
    if p_prev > 0:
        # add note
        duration_sec = duration * hop / float(fs)
        onset_sec = onset * hop / float(fs)
        notes.append((onset_sec, duration_sec, p_prev))

    return notes


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


def audio_to_midi_melodia(outfile, bpm, smooth=0.15, minduration=0.1,
                          savejams=False):

    # define analysis parameters
    fs = sr
    hop = hop_length
    data = y



    # extract melody using melodia vamp plugin
    print("Extracting melody f0 with MELODIA...")
    melody = vamp.collect(data, sr, "mtg-melodia:melodia",
                          parameters={"voicing": 0.2})

    # hop = melody['vector'][0]
    pitch = melody['vector'][1]

    print("pitch is: ", pitch)
    print("and f0s are : ", f0s)

    # impute missing 0's to compensate for starting timestamp
    pitch = np.insert(pitch, 0, [0]*8)

    # debug
    # np.asarray(pitch).dump('f0.npy')
    # print(len(pitch))

    # convert f0 to midi notes
    print("Converting Hz to MIDI notes...")
    midi_pitch = hz2midi(pitch)

    # segment sequence into individual midi notes
    notes = midi_to_notes(midi_pitch, fs, hop, smooth, minduration)

    # save note sequence to a midi file
    print("Saving MIDI to disk...")
    save_midi(outfile, notes, bpm)

    if savejams:
        print("Saving JAMS to disk...")
        jamsfile = outfile.replace(".mid", ".jams")
        track_duration = len(data) / float(fs)
        save_jams(jamsfile, notes, track_duration, os.path.basename(infile))

    print("Conversion complete.")


#audio_to_midi_melodia(y, sr, hop, outfile, bpm, smooth=0.25, minduration=0.1, savejams=False)
# audio_to_midi_melodia(y,sr, hop_length, "/home/ashan/Desktop/test_midi.mid")
