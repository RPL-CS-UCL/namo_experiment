import numpy as np

data_directory = 'saved_data'
filename = data_directory + '/data_09-05-18-41.npy'

with open(filename, 'rb') as f:
    print(f)
    a = np.load(f)



print(a)