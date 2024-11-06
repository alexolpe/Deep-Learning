from transformers import BertForQuestionAnswering
from transformers import TrainingArguments
import pickle
from transformers import Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from transformers import pipeline

model_name = 'bert-base-uncased'
model = BertForQuestionAnswering.from_pretrained(model_name)

training_args = TrainingArguments (
    output_dir ='./results', # output directory
    use_mps_device = False ,
    num_train_epochs =5 , # total number of training epochs , change this as you need
    per_device_train_batch_size=8 , # batch size per device during training ,change this as you need
    per_device_eval_batch_size=8 , # batch size for evaluation , change this as you need
    weight_decay =0.01 , # strength of weight decay
    logging_dir ='./logs', # directory for storing logs
    logging_strategy="epoch",
    save_strategy="epoch"
)

with open('/home/aolivepe/ECE60146/HW10/dataset/train_dict.pkl','rb') as f:
    train_dict = pickle.load(f)
with open('/home/aolivepe/ECE60146/HW10/dataset/test_dict.pkl','rb') as f:
    test_dict = pickle.load(f)
with open('/home/aolivepe/ECE60146/HW10/dataset/eval_dict.pkl','rb') as f:
    eval_dict = pickle.load(f)
with open('/home/aolivepe/ECE60146/HW10/dataset/train_data_processed.pkl','rb') as f:
    train_processed = pickle.load(f)
with open('/home/aolivepe/ECE60146/HW10/dataset/test_data_processed.pkl','rb') as f:
    test_processed = pickle.load(f)
with open('/home/aolivepe/ECE60146/HW10/dataset/eval_data_processed.pkl','rb') as f:
    eval_processed = pickle.load(f)
    
train_dataset = Dataset.from_pandas(pd.DataFrame(train_processed))
eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_processed))
test_dataset = Dataset.from_pandas(pd.DataFrame(test_processed))
trainer = Trainer(
    model=model , # the instantiated Transformers model to be fine - tuned
    args = training_args , # training arguments , defined above
    train_dataset = train_dataset , # training dataset
    eval_dataset = eval_dataset # evaluation dataset
)

trainer.train()

## QUALITATIVE AND QUANTITATIVE EVALUATION ------------------------------------------------------------------------
def compute_exact_match(prediction, truth):
    return int(prediction == truth)

def f1_score(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    # if either the prediction or the truth is no - answer then f1 = 1 if they agree , 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = set(pred_tokens) & set(truth_tokens)
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    prec = len(common_tokens)/len(pred_tokens)
    rec = len(common_tokens)/len(truth_tokens)
    return 2*(prec*rec)/(prec+rec)

x = trainer.predict(test_dataset)
start_pos, end_pos = x.predictions
start_pos = np.argmax(start_pos, axis =1)
end_pos = np.argmax(end_pos, axis =1)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Arrays where we store Exact Match Score and F1 Score for each question
em = []
f1 = []

for k, (i, j) in enumerate(zip(start_pos, end_pos)):
    tokens = tokenizer.convert_ids_to_tokens(test_processed['input_ids'][k])
    print('Question :', test_dict['question'][k])
    print('Answer : ',''.join(tokens[i:j+1]))
    print('Correct Answer : ', test_dict['answers'][k]['text'][0])
    print('Exact Match : ', compute_exact_match(''.join(tokens[i:j+1]), test_dict['answers'][k]['text'][0]))
    print('F1 Score : ', f1_score(''.join(tokens[i:j+1]), test_dict['answers'][k]['text'][0]))
    print('---')
    em.append(compute_exact_match(''.join(tokens[i:j+1]), test_dict['answers'][k]['text'][0]))
    f1.append(f1_score(''.join(tokens[i:j+1]), test_dict['answers'][k]['text'][0]))
    
def compute_average(arr):
    total = sum(arr)
    return total / len(arr)

def compute_median(arr):
    sorted_arr = sorted(arr)
    n = len(arr)
    if n % 2 == 0:
        median = (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2
    else:
        median = sorted_arr[n // 2]
    return median

# Compute average
average_em = compute_average(em)
print("Average Exact Match:", average_em)

# Compute median
median_em = compute_median(em)
print("Median Exact Match:", median_em)

# Compute average
average_f1 = compute_average(f1)
print("Average F1 Score:", average_f1)

# Compute median
median_f1 = compute_median(f1)
print("Median F1 Score:", median_f1)

## COMPARISON ---------------------------------------------------------------------------------------------
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

# Arrays where we store Exact Match Score and F1 Score for each question
em = []
f1 = []

for i in range(len(test_dict['question'])):
    result = question_answerer(question = test_dict['question'][i], context = test_dict['context'][i])
    print('Question : ', test_dict['question'][i])
    print('Answer : ', result['answer'])
    print('Correct Answer : ', test_dict['answers'][i]['text'][0])
    print('Exact Match : ', compute_exact_match(result['answer'],test_dict['answers'][i]['text'][0]))
    print('F1 Score : ', f1_score(result['answer'], test_dict['answers'][i]['text'][0]))
    print('---')
    em.append(compute_exact_match(result['answer'],test_dict['answers'][i]['text'][0]))
    f1.append(f1_score(result['answer'], test_dict['answers'][i]['text'][0]))

# Compute average
average_em = compute_average(em)
print("Average Exact Match:", average_em)

# Compute median
median_em = compute_median(em)
print("Median Exact Match:", median_em)

# Compute average
average_f1 = compute_average(f1)
print("Average F1 Score:", average_f1)

# Compute median
median_f1 = compute_median(f1)
print("Median F1 Score:", median_f1)