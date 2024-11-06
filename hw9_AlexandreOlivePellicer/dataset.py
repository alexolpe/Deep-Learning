import torch
import sys
import random
import pickle
import csv
from transformers import DistilBertModel
from transformers import DistilBertTokenizer
import math

## Read csv -------------------------------------------
sentences = []
sentiments = []
count = 0
with open('data.csv', 'r') as f:
    reader =csv.reader( f )
    next(reader)
    for row in reader :
        count += 1
        sentences.append(row[0])
        sentiments.append(row[1])
        
print(sentences)
print(sentiments)


## Get one-hot vectors for the sentiments -------------------------
def sentiment_to_tensor(sentiment):      
    sentiment_tensor = torch.zeros(3)
    if sentiment == "positive":
        sentiment_tensor[0] = 1
    elif sentiment == "neutral": 
        sentiment_tensor[1] = 1
    elif sentiment == "negative": 
        sentiment_tensor[2] = 1
    sentiment_tensor = sentiment_tensor.type(torch.long)
    return sentiment_tensor

encoded_sentiments = []
for sentiment in sentiments:
    encoded_sentiments.append(sentiment_to_tensor(sentiment))
    
## Word level tokenization ------------------------------------------------
word_tokenized_sentences = [ sentence . split () for sentence in sentences ]

max_len = max([len(sentence) for sentence in word_tokenized_sentences])
padded_sentences = [sentence + ['[PAD]'] * (max_len - len(sentence)) for sentence in word_tokenized_sentences]

vocab = {}
vocab ['[PAD]'] = 0
for sentence in padded_sentences :
    for token in sentence :
        if token not in vocab :
            vocab[token] = len(vocab)
# convert the tokens to ids
padded_sentences_ids = [[vocab[token] for token in sentence ] for sentence in padded_sentences]

## Subord level tokenization ------------------------------------------------
model_ckpt = "distilbert-base-uncased"
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

bert_tokenized_sentences_ids = [distilbert_tokenizer.encode(sentence, padding ='max_length', truncation =True, max_length = max_len) for sentence in sentences ]

bert_tokenized_sentences_tokens = [distilbert_tokenizer.convert_ids_to_tokens(sentence) for sentence in bert_tokenized_sentences_ids]

##Extracting embeddings -------------------------------------------------------
model_name = 'distilbert/distilbert-base-uncased'
distilbert_model = DistilBertModel.from_pretrained(model_name)
# extract word embeddings
# we will use the last hidden state of the model
# you can use the other hidden states if you want
# the last hidden state is the output of the model
# after passing the input through the model

word_embeddings = []
# convert padded sentence tokens into ids
for tokens in padded_sentences_ids :
    input_ids = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        outputs = distilbert_model(input_ids)
    word_embeddings.append(outputs.last_hidden_state)
print(word_embeddings[0].shape)

# subword embeddings extraction
subword_embeddings = []
for tokens in bert_tokenized_sentences_ids:
    input_ids = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        outputs = distilbert_model(input_ids)
    subword_embeddings.append(outputs.last_hidden_state)
print(subword_embeddings[1].shape)

## Saving embeddings and one-hot vectors in training and testing dictionaries ------------------
dictionary = {}
for i in range(len(subword_embeddings)):
    dictionary[i] = {}
    dictionary[i]["sentiment"] = encoded_sentiments[i]
    dictionary[i]["word_embeddings"] = word_embeddings[i]
    dictionary[i]["subword_embeddings"] = subword_embeddings[i]
    
# Calculate the number of keys for each dictionary
total_keys = len(dictionary)
first_dict_keys = math.ceil(0.8 * total_keys)  # 80% of total keys
second_dict_keys = total_keys - first_dict_keys  # Remaining 20% of total keys

# Create two new dictionaries
first_dictionary = {}
second_dictionary = {}

# Populate the first dictionary with 80% of the keys
for i in range(first_dict_keys):
    first_dictionary[i] = dictionary[i]

# Populate the second dictionary with the remaining 20% of the keys
for i in range(first_dict_keys, total_keys):
    second_dictionary[i - first_dict_keys] = dictionary[i]

with open('training_dictionary.pickle', 'wb') as f:
    pickle.dump(first_dictionary, f)

with open('testing_dictionary.pickle', 'wb') as f:
    pickle.dump(second_dictionary, f)