import struct
import numpy as np

with open('train_data/train-images.idx3-ubyte', 'rb') as training_data_file:
    magic_numer = struct.unpack('>I', training_data_file.read(4))[0]
    number_of_images = struct.unpack('>I', training_data_file.read(4))[0]
    number_of_rows = struct.unpack('>I', training_data_file.read(4))[0]
    number_of_columns = struct.unpack('>I', training_data_file.read(4))[0]
    training_data = np.frombuffer(training_data_file.read(), dtype=np.uint8)
    training_data = training_data.reshape((number_of_images, number_of_rows, number_of_columns))

with open('train_data/train-labels.idx1-ubyte', 'rb') as training_label_file:
    magic_numer = struct.unpack('>I', training_label_file.read(4))[0]
    number_of_lables = struct.unpack('>I', training_label_file.read(4))[0]
    training_label = np.frombuffer(training_label_file.read(), dtype=np.uint8)


with open('test_data/t10k-images.idx3-ubyte', 'rb') as testing_data_files:
    magic_number = struct.unpack('>I', testing_data_files.read(4))[0]
    number_of_images = struct.unpack('>I', testing_data_files.read(4))[0]
    number_of_rows = struct.unpack('>I', testing_data_files.read(4))[0]
    number_of_columns = struct.unpack('>I', testing_data_files.read(4))[0]
    testing_data = np.frombuffer(testing_data_files.read(), dtype=np.uint8)
    testing_data = testing_data.reshape((number_of_images,number_of_rows,number_of_columns))

with open('test_data/t10k-labels.idx1-ubyte', 'rb') as testing_label_file:
    magic_number = struct.unpack('>I', testing_label_file.read(4))[0]
    number_of_labels = struct.unpack('>I', testing_label_file.read(4))[0]
    testing_label = np.frombuffer(testing_label_file.read(), dtype=np.uint8)  

training_data = training_data / 255.0
testing_data = testing_data / 255.0

print("Training data shape:", training_data.shape)
print("Training label shape:", training_label.shape)
print("Testing data shape:", testing_data.shape)        
print("Testing label shape:", testing_label.shape)

