"""Convert raw GloVe word vector text file to h5."""
import h5py
import numpy as np
import gc

# """Convert raw GloVe word vector text file to h5."""
glove_vectors = []
vocab = []
with open('models/embeddings/glove.840B.300d.txt', 'r',encoding="utf-8") as f:
    for line in f:
        if(len(line.strip().split()[1:]) > 300 ):
            glove_vectors+=[float(val) for val in line.strip().split()[(len(line.strip().split()[0:])-300):]]
            # tmp = line.strip().split()[0:(len(line.strip().split()[0:])-300)]
            # s = " "
            # str1 = s.join(tmp) +"\n"
            # vocab.append(str1)
            # print("exceed 1 word: ", str1)
        else:
            glove_vectors+=[float(val) for val in line.strip().split()[1:]]
            # str1 = line.strip().split()[0] +"\n"
            # vocab.append(str1)

        # except ValueError:
        #     print("error on line",len(line.strip().split()[1:]))
glove_vectors = np.array(glove_vectors).astype(np.float32)       
f = h5py.File('glove.840B.300d.h5', 'w')
f.create_dataset(data=glove_vectors, name='embedding')
print('embedding dataset created.')
f.close()
del glove_vectors
gc.collect()

with open('models/embeddings/glove.840B.300d.txt', 'r',encoding="utf-8") as f:
    for line in f:
        if(len(line.strip().split()[1:]) > 300 ):
            # glove_vectors+=[float(val) for val in line.strip().split()[(len(line.strip().split()[0:])-300):]]
            tmp = line.strip().split()[0:(len(line.strip().split()[0:])-300)]
            s = " "
            str1 = s.join(tmp)
            vocab.append(str1)
            print("exceed 1 word: ", str1)
        else:
            # glove_vectors+=[float(val) for val in line.strip().split()[1:]]
            str1 = line.strip().split()[0]
            vocab.append(str1)

f = h5py.File('glove.840B.300d.h5', 'r+')
vocab = '\n'.join(vocab)
f.create_dataset(data=vocab, name='words_flatten')
print('words_flatten dataset created.')
f.close()