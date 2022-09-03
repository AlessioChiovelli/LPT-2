import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from time import sleep
from transformers import Trainer,TrainingArguments,DataCollatorWithPadding

#Creating the csv for training,test and validation
def shuffle_and_split_dataframes():
    data = pd.read_csv("./data/cyberbullying_tweets.csv")
    # """SEE THE CLASSES"""
    # print(self.data.cyberbullying_type.unique())
    data = shuffle(data)
    train_len = int(data.shape[0]*0.7)
 
    #Splitting in Train, Validation and Test Set
    data_train, data_test = train_test_split(data,train_size = train_len,test_size = data.shape[0] - train_len)
    data_test, data_validation = train_test_split(data_test,train_size = int(0.5 * data_test.shape[0]),test_size = int(0.5 * data_test.shape[0]))
    data_train.to_csv("./data/train_data.csv",index = False)
    data_test.to_csv("./data/test_data.csv",index = False)
    data_validation.to_csv("./data/validation_data.csv",index = False)

'''
Functions to map to the dataset for handling properly the text
'''
def lowerWords(phrase):return phrase.lower()
def tokenize_sample(example):return tokenizer(example["tweet_text"], padding = "max_length", truncation = True, max_length = 512, return_tensors="pt")
def create_target_tensors(type_bullying):
    if "age" == type_bullying: return [1,0,0,0,0,0]
    if "ethnicity" == type_bullying: return [0,1,0,0,0,0]
    if "gender" == type_bullying: return [0,0,1,0,0,0]
    if "not_cyberbullying" == type_bullying: return [0,0,0,1,0,0]
    if "other_cyberbullying" == type_bullying: return [0,0,0,0,1,0]
    if "religion" == type_bullying: return [0,0,0,0,0,1]

class MyPytorchDataset(Dataset):
    def __init__(self, csv, tokenizer):

        tokenizer = tokenizer
        dataset_as_df = pd.read_csv(csv)

        input_ids = []
        token_type_ids = []
        attention_mask = []
        labels = []
        
        self.len_dataset = dataset_as_df.shape[0]
        for row in range(self.len_dataset):
            
            tokenized_sample = tokenizer(dataset_as_df.loc[row].at["tweet_text"], padding = "max_length", truncation = True, max_length = 512)
            input_ids.append(tokenized_sample["input_ids"])
            token_type_ids.append(tokenized_sample["token_type_ids"])
            attention_mask.append(tokenized_sample["attention_mask"])
            
            labels.append(create_target_tensors(dataset_as_df.loc[row].at["cyberbullying_type"]))
        
        self.input_ids = torch.tensor(input_ids,dtype = torch.int32)
        self.token_type_ids = torch.tensor(token_type_ids,dtype = torch.int32)
        self.attention_mask = torch.tensor(attention_mask,dtype = torch.int32)    
        self.y = torch.tensor(labels,dtype = torch.float32)
    
    def __len__(self):return self.len_dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.attention_mask[idx], self.y[idx]

if __name__ == "__main__":

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = "microsoft/MiniLM-L12-H384-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation = True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.classifier = nn.Sequential(
        nn.Linear(384,6),
        nn.Softmax(dim = -1)
    )
    try:
        model.load_state_dict(torch.load("ProgettoSupervised/Microsoft_MiniLM-L12-H384-Fine_Tuned_Over_KaggleCyberbullying_Classifier.pt", map_location=device))
        print('Trained Network Found!')
    except:
        print('No Trained Network Found!\n\nStarting a new Fine Tune')
    model.to(device)

    train_path,validation_path,test_path = "./data/train_data.csv","./data/validation_data.csv","./data/test_data.csv"

    lr = 2e-5
    epochs = 2
    s_mini_batch = 8
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    CE_loss = torch.nn.CrossEntropyLoss()

    labels_of_bullying = {'age' : 0, 'ethnicity' : 1, 'gender' : 2, 'not_cyberbullying' : 3, 'other_cyberbullying' : 4, 'religion' : 5}
    keys_of_bullying = list(labels_of_bullying.keys())
    datasets = {"train":train_path,"validation":validation_path,"test":test_path}
    train_dataset = MyPytorchDataset(datasets["train"], tokenizer)
    validation_dataset = MyPytorchDataset(datasets["validation"], tokenizer)
    test_dataset = MyPytorchDataset(datasets["test"], tokenizer)
    
    shuffle_and_split_dataframes()
    train_loader = DataLoader(train_dataset, batch_size = s_mini_batch)

    '''
    WITH PYTORCH API
    '''
    model.train()
    # accuracies_over_epochs = []
    num_mini_batch = (len(train_dataset)-1)//s_mini_batch + 1

    torch.save(model.state_dict(), "ProgettoSupervised/Microsoft_MiniLM-L12-H384-preSave.pt")
    print('Pre save Done!')
    sleep(3)
    for epoch in range(epochs):
        right_training = 0
        for id,batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            
            input_ids = batch[0]
            token_type_ids = batch[1]
            attention_mask = batch[2]
            labels = batch[3]
            
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels =  labels.to(device)
            outputs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
            # print(outputs)
            # sleep(5)
            
            loss = CE_loss(outputs['logits'],labels)
            loss.backward()
            optimizer.step()
            
            batch_predictions_training = []
            right_training_mini_batch = 0
            for i_th_sample in range(s_mini_batch):
                classes_output = outputs.logits.detach().clone()
                class_predicted = torch.argmax(classes_output[i_th_sample])
                real_class = torch.argmax(labels[i_th_sample])
                print('class_predicted:{}   ///   real_class:{}\n'.format(class_predicted,real_class))
                if class_predicted == real_class:
                    # batch_predictions_training.append(1)
                    right_training_mini_batch += 1
                # else:batch_predictions_training.append(0)
            # correct_predictions_training.append(batch_predictions_training)
            right_training += right_training_mini_batch

            precision_over_mini_batch = right_training_mini_batch / s_mini_batch
            
            print('epoch: {} -- {}/{} batches   \nloss : {}  //  precision over mini batch : {}'.format(epoch,id,num_mini_batch,loss,precision_over_mini_batch))
        print('Accuracy over {}-th epoch : {} / {}'.format(epoch,right_training,len(train_dataset)))

    # accuracies_over_epochs(right_training / len(train_dataset))

    torch.save(model.state_dict(), "ProgettoSupervised/Microsoft_MiniLM-L12-H384-Fine_Tuned_Over_KaggleCyberbullying_Classifier.pt")            

#     '''
#     WITH HUGGINFACE API
#     '''

#     data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
#     train_args = TrainingArguments(
#         output_dir = "./trainingResults",
#         num_train_epochs = 2,
#         learning_rate= 5e-5,
#         per_device_train_batch_size = 32,
#         per_device_eval_batch_size = 32
#     )
    
#     print(train_dataset[0])
#     # trainer = Trainer(
#     #     model,
#     #     train_args,
#     #     train_dataset=train_dataset,
#     #     eval_dataset=validation_dataset,
#     #     tokenizer=tokenizer,
#     #     data_collator = data_collator
#     # )
#     # trainer.train()
    
# """
# !!ERROR

#         # If we have a list of dicts, let's convert it in a dict of lists
#         # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
#         if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
#             encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

#         # The model's main input name, usually `input_ids`, has be passed for padding
#         if self.model_input_names[0] not in encoded_inputs:
#             raise ValueError(
#                 "You should supply an encoding or a list of encodings to this method "
#                 f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
#             )

#         required_input = encoded_inputs[self.model_input_names[0]]
# """
