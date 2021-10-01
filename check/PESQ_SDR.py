from pypesq import pesq
import wave


def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    # The dtype depends on the value of pulse-code modulation. The int16 is set for 16-bit PCM.
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

clean_file = ""
noise_file = ""

clean_wav = wave.open(clean_file, "r")
noise_wav = wave.open(noise_file, "r")
clean = cal_amp(clean_wav)
noise = cal_amp(noise_wav)

print(pesq(16000, clean, noise, 'wb'))