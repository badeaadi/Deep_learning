import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import re
import gc


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

count = 3

DROPOUT_RATE = 60

for i in range(count * 100, min((count + 1) * 100, len(records))):
    print(i)
    seiz = False

    print(records[i])

    if (records[i] in seizure_records):
        seiz = True

    record = records[i]
    f = pyedflib.EdfReader("data/" + record)

    n = f.signals_in_file
    
    if n < 23:
        continue
    
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)

    sigbufs_reshaped = np.reshape(sigbufs, (sigbufs.shape[0], -1, 256))
    
    nonseizure_count = int(np.random.rand(1) * 100)

    if seiz == False:
        for j in range(sigbufs_reshaped.shape[1]):
            if nonseizure_count % DROPOUT_RATE == 0:
                nonseizure_bufs.append(sigbufs_reshaped[:23, j, :])

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
                    seizure_bufs.append(sigbufs_reshaped[:23, j, :])
                    ok = True
                    break

            if ok == False:
                if nonseizure_count % DROPOUT_RATE == 0:
                    nonseizure_bufs.append(sigbufs_reshaped[:23, j, :])

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

np.savez("signals" + str(count) + ".npz", signals=bufs, labels=bufs_labels)







def concatenate_datasets(a : int, b : int):
    
    all_signals = np.empty((0, 23, 256))
    all_labels = np.empty((0))
    
    for count in range(a, b):
        
        print('{}th dataset'.format(count))
        input_file = "signals" + str(count) + ".npz"
        npz_file = np.load(input_file, allow_pickle=True)   
        
        signals = npz_file['signals']
        labels = npz_file['labels']
        
        print(signals.shape)
        print(labels.shape)
            
        all_signals = np.concatenate((all_signals, signals), axis = 0)
        all_labels = np.concatenate((all_labels, labels), axis = 0)
    
    print('Final shape : {}'.format(all_signals.shape))
    print('Final labels shape : {}'.format(all_labels.shape))
    
    np.savez("final_signals.npz", signals=all_signals, labels=all_labels)
    
concatenate_datasets(0, 7)