from load_data import training_data , training_label
from network import forward_pass,cross_entropy_loss,backpropogation, initialize_network

W1, W2, b1, b2 = initialize_network(784, 128, 10)
learning_rate = 0.01
for epoch in range(20):
    total_loss = 0 
    for image, label in zip(training_data ,training_label):
        output, hidden, hidden_raw  = forward_pass(image,W1,W2,b1,b2)
        loss = cross_entropy_loss(output, label)
        W1,W2,b1,b2 =backpropogation(input=image, hidden=hidden,hidden_raw=hidden_raw,output=output,label=label,W2=W2,W1=W1,b1=b1,b2=b2,learning_rate=learning_rate)
        total_loss += loss
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(training_data)}")
