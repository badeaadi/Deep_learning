import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import re
import gc

# file_name = "../chbmit-1.0.0.physionet.org/chb01/chb01_03.edf"
# f = pyedflib.EdfReader(file_name)

# n = f.signals_in_file
# signal_labels = f.getSignalLabels()
# sigbufs = np.zeros((n, f.getNSamples()[0]))
# for i in np.arange(n):
#     sigbufs[i, :] = f.readSignal(i)

# print(sigbufs)
# print(signal_labels)
# print(n)
# print(f.datarecords_in_file)
# print(f.datarecord_duration)

# mfcc = librosa.stft(sigbufs[20], n_fft=8192,
#                     win_length=2048, center=True)
# xstft = librosa.amplitude_to_db(abs(mfcc))

# librosa.display.specshow(xstft, x_axis='time', sr=256)
# plt.colorbar()
# plt.show()

gc.enable()

records = []
with open("data/RECORDS") as file:
    lines = file.readlines()

    for line in lines:
        records.append(line.strip("\n"))

seizure_records = []
with open("data/RECORDS-WITH-SEIZURES") as file:
    lines = file.readlines()

    for line in lines:
        seizure_records.append(line.strip("\n"))

seizures = []

for file_idx in range(1, 25):
    file_name = f"data/chb{str(file_idx) if file_idx >= 10 else ('0' + str(file_idx))}/chb{str(file_idx) if file_idx >= 10 else ('0' + str(file_idx))}-summary.txt"

    print(file_name)
    file = open(file_name)
    lines = file.readlines()

    i = 0

    while i < len(lines):
        if ".edf" not in lines[i]:
            i += 1
            continue

        filenm = lines[i].strip("\n").split(" ")[2]

        while "Number of Seizures" not in lines[i]:
            i += 1

        num_seizures = int(lines[i].strip("\n").split(" ")[5])

        for seiz in range(num_seizures):
            i += 1
            start_time = re.sub("\s\s+", " ", lines[i].strip("\n")).split(" ")
            start_time = int(start_time[3 if len(start_time) == 5 else 4])

            i += 1
            end_time = re.sub("\s\s+", " ", lines[i].strip("\n")).split(" ")
            end_time = int(end_time[3 if len(end_time) == 5 else 4])

            seizures.append((filenm, start_time, end_time))

nonseizure_bufs = []
seizure_bufs = []


# print(records.index("chb04/chb04_28.edf"))

count = 0

for i in range(count * 100, min((count + 1) * 100, len(records))):
    if i % 10 == 0:
        gc.collect()
        
    print(i)
    seiz = False
    
    print(records[i])
    
    if (records[i] in seizure_records):
        seiz = True
        print('Seizure')

    record = records[i]
    f = pyedflib.EdfReader("data/" + record)
    
    n = f.signals_in_file

    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)

    sigbufs_reshaped = np.reshape(sigbufs, (sigbufs.shape[0], -1, 256))
    nonseizure_count = 0
    
    if seiz == False:
        for j in range(sigbufs_reshaped.shape[1]):
            if nonseizure_count % 500 == 0:
                nonseizure_bufs.append(sigbufs_reshaped[:, j, :])

            nonseizure_count += 1
    else:
        seizs = []
        for s in seizures:
            if s[0] in record:
                seizs.append(s)

        for j in range(sigbufs_reshaped.shape[1]):
         
            ok = False
            for s in seizs:
                if j >= s[1] and j <= s[2]:
                    seizure_bufs.append(sigbufs_reshaped[:, j, :])
                    ok = True
                    break

            if ok == False:
                if nonseizure_count % 500 == 0:
                    nonseizure_bufs.append(sigbufs_reshaped[:, j, :])
                    

                nonseizure_count += 1

nonseizure_bufs = np.array(nonseizure_bufs)

print(nonseizure_bufs.shape)

seizure_bufs = np.array(seizure_bufs)


print(seizure_bufs.shape)

bufs = np.concatenate((nonseizure_bufs, seizure_bufs), axis=0)

print(bufs.shape)

bufs_labels = np.concatenate(
    (np.zeros(nonseizure_bufs.shape[0]), np.ones(seizure_bufs.shape[0])), axis=0)

print(bufs_labels.shape)

mfccs = []

'''
for channel in range(0, 23):

        mfcc = librosa.feature.mfcc(y=bufs[i:i, channel, :].flatten(), sr=256, n_mfcc=32)
        print(mfcc.shape)
    
        mfccs.append(mfcc)

mfccs = np.array(mfccs)
print(mfccs.shape)
'''

np.savez("signals" + str(count) + ".npz", signals=bufs, labels=bufs_labels)
# bufs = []

# for i in range(0, 2):
#     record = records[i]
#     f = pyedflib.EdfReader(record)

#     n = f.signals_in_file

#     sigbufs = np.zeros((n, f.getNSamples()[0]))
#     for i in np.arange(n):
#         sigbufs[i, :] = f.readSignal(i)

#     bufs.append(sigbufs)
