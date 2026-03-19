import numpy as np
from load_data import testing_data , testing_label
from network import forward_pass

W1 = np.load('W1.npy')
W2 = np.load('W2.npy')
b1 = np.load('b1.npy')
b2 = np.load('b2.npy')

print("Began Testing")
count = 0
for image , label in zip(testing_data, testing_label):
    output=forward_pass(image,W1,W2,b1,b2)[0]
    if np.argmax(output) == label:
        count += 1
    
accuracy = (count/10000)*100


print(f"Accuracy: {accuracy:.2f}%")