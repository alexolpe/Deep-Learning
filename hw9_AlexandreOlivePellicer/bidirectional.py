import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle

device = "cuda:0"

## Dataset class -----------------------------------------------
class SentimentDataset(Dataset):
    def __init__(self, train_or_test, word_or_subword):
        self.train_or_test = train_or_test
        self.word_or_subword = word_or_subword
        
        with open('training_dictionary.pickle', 'rb') as f:
            self.training_dictionary = pickle.load(f)

        # Load the second dictionary from the pickle file
        with open('testing_dictionary.pickle', 'rb') as f:
            self.testing_dictionary = pickle.load(f)


    def __len__(self):
        length = 0
        if self.train_or_test == "train":
            length = len(self.training_dictionary)
        elif self.train_or_test == "test":
            length = len(self.testing_dictionary)
        return length

    def __getitem__(self, i):
        if self.train_or_test == "train":
            if self.word_or_subword == "word":
                sentence = self.training_dictionary[i]["word_embeddings"]
                sentence = sentence.permute(1, 0, 2)
            elif self.word_or_subword == "subword":
                sentence = self.training_dictionary[i]["subword_embeddings"]
                sentence = sentence.permute(1, 0, 2)
            label = self.training_dictionary[i]["sentiment"]
            
        elif self.train_or_test == "test":
            if self.word_or_subword == "word":
                sentence = self.testing_dictionary[i]["word_embeddings"]
                sentence = sentence.permute(1, 0, 2)
            elif self.word_or_subword == "subword":
                sentence = self.testing_dictionary[i]["subword_embeddings"]
                sentence = sentence.permute(1, 0, 2)
            label = self.testing_dictionary[i]["sentiment"]
        return sentence, label.type(torch.float)

# GRU model (based on DLStudio implementation)-----------------------------------------------------------
class SentimentGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3): 
        super(SentimentGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = out[:,:,:128]
        out = self.fc(self.relu(out[:,-1]))
        out = self.logsoftmax(out)
        return out, h
    
    def init_hidden(self):
        weight = next(self.parameters()).data
        #                  num_layers  batch_size    hidden_size
        hidden = weight.new(  6,          1,         self.hidden_size    ).zero_()
        return hidden

## Training loss
criterion = nn.NLLLoss()

## Optimizer
learning_rate = 0.001
model = SentimentGRU(input_size=768, hidden_size=128, output_size=3).to(device) # 3 output classes
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.squeeze(0).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            hidden = model.init_hidden().to(device)
            for k in range(inputs.shape[1]):
                output, hidden = model(torch.unsqueeze(torch.unsqueeze(inputs[0,k],0),0), hidden)
            loss = criterion(output, torch.argmax(labels, 1))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
    
    # Save generator model and loss plot
    torch.save(model.state_dict(), './model_bidirectional_word_100_epochs.pth')
    
    dictionary_losses = {}

    nombre_imagen = 'yes'
    dictionary_losses[nombre_imagen] = {
        'criterion1': losses
    }
    
    with open('/home/aolivepe/ECE60146/HW9/loss_bidirectional_word_100_epochs.pkl', 'wb') as archivo:
        pickle.dump(dictionary_losses, archivo)

    # Plot training loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.savefig('training_loss_plot.png')
    plt.show()

## Evaluate your model
def evaluate_model(model, dataloader):
    classification_accuracy = 0.0
    negative_total = 0
    positive_total = 0
    neutral_total = 0
    confusion_matrix = torch.zeros(3,3)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.squeeze(0).to(device)
            labels = labels.to(device)
            
            hidden = model.init_hidden().to(device)
            for k in range(inputs.shape[1]):
                output, hidden = model(torch.unsqueeze(torch.unsqueeze(inputs[0,k],0),0), hidden)
            predicted_idx = torch.argmax(output).item()
            gt_idx = torch.argmax(labels).item()
            if i % 50 == 49:
                print("   [i=%d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
            if predicted_idx == gt_idx:
                classification_accuracy += 1
            if gt_idx == 0: 
                positive_total += 1
            elif gt_idx == 1:
                neutral_total += 1
            elif gt_idx == 2:
                negative_total += 1
            confusion_matrix[gt_idx,predicted_idx] += 1
    print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
    out_percent = np.zeros((3,3), dtype='float')
    out_percent[0, 0] = "%.3f" % (100 * confusion_matrix[0, 0] / float(positive_total))
    out_percent[0, 1] = "%.3f" % (100 * confusion_matrix[0, 1] / float(positive_total))
    out_percent[0, 2] = "%.3f" % (100 * confusion_matrix[0, 2] / float(positive_total))
    out_percent[1, 0] = "%.3f" % (100 * confusion_matrix[1, 0] / float(neutral_total))
    out_percent[1, 1] = "%.3f" % (100 * confusion_matrix[1, 1] / float(neutral_total))
    out_percent[1, 2] = "%.3f" % (100 * confusion_matrix[1, 2] / float(neutral_total))
    out_percent[2, 0] = "%.3f" % (100 * confusion_matrix[2, 0] / float(negative_total))
    out_percent[2, 1] = "%.3f" % (100 * confusion_matrix[2, 1] / float(negative_total))
    out_percent[2, 2] = "%.3f" % (100 * confusion_matrix[2, 2] / float(negative_total))
    print("\n\nNumber of positive reviews tested: %d" % positive_total)
    print("\n\nNumber of neutral reviews tested: %d" % neutral_total)
    print("\n\nNumber of negative reviews tested: %d" % negative_total)
    print("\n\nDisplaying the confusion matrix:\n")
    out_str = "                      "
    out_str +=  "%18s  %18s  %18s" % ('predicted positive', 'predicted neutral', 'predicted negative')
    print(out_str + "\n")
    for i,label in enumerate(['true positive', 'true neutral', 'true negative']):
        out_str = "%12s:  " % label
        for j in range(3):
            out_str +=  "%18s%%" % out_percent[i,j]
        print(out_str)


## Define datasets and dataloaders
train_dataset =SentimentDataset("train", "word")
test_dataset =SentimentDataset("test", "word")

train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True )
test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True )

## Run training and evaluation
train_model(model, train_dataloader, criterion, optimizer, num_epochs=100)
evaluate_model(model, test_dataloader)
