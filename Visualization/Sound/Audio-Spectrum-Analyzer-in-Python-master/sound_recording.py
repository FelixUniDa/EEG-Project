# https://realpython.com/playing-and-recording-sound-python/

import sounddevice
from scipy.io.wavfile import write

fs = 44100
second = 3
print("recording...")
record_voice = sounddevice.rec(int(second * fs), samplerate=fs, channels=1)
sounddevice.wait()
write("water_my_friend4.wav", fs, record_voice)
