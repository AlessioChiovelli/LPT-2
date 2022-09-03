from PytorchTraining import MyPytorchDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from time import sleep
from tqdm import tqdm
import random



if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    # else:
    #     try:
    #         device = torch.device("mps")
    #     except:
    #         device = torch.device('cpu')
    # device = torch.device("mps")
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    sleep(3)
    checkpoint = "microsoft/MiniLM-L12-H384-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation = True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.classifier = nn.Sequential(
        nn.Linear(384,6),
        nn.Softmax(dim = -1)
    )
    try:
        model.load_state_dict(torch.load("risultati/Microsoft_MiniLM-L12-H384-Fine_Tuned_Over_KaggleCyberbullying_Classifier.pt", map_location=device))
        print('Trained Network Found!')
    except:
        print('No Trained Network Found!\n\nStarting a new Fine Tune')
        sleep(10)
    model.to(device)

    s_mini_batch = 8

    labels_of_bullying = {'age' : 0, 'ethnicity' : 1, 'gender' : 2, 'not_cyberbullying' : 3, 'other_cyberbullying' : 4, 'religion' : 5}
    train_path,validation_path,test_path = "data/train_data.csv","data/validation_data.csv","data/test_data.csv"
    datasets = {"train":train_path,"validation":validation_path,"test":test_path}
    validation_dataset = MyPytorchDataset(datasets["validation"], tokenizer)
    test_dataset = MyPytorchDataset(datasets["test"], tokenizer)
    def evaluate_over(set_dataloader,name_for_save):
        wrong_predictions_to_analize = dict()
        right = 0
        samples = 0
        with torch.no_grad():
            for batch in tqdm(set_dataloader):
                samples += s_mini_batch
                input_ids = batch[0]
                token_type_ids = batch[1]
                attention_mask = batch[2]
                labels = batch[3]
                
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels =  labels.to(device)
                
                outputs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
                rights_of_batch = 0
                for i_th_sample in range(s_mini_batch):
                    try:
                        class_predicted = torch.argmax(outputs.logits[i_th_sample])
                        real_class = torch.argmax(labels[i_th_sample])
                        if class_predicted == real_class:
                            right += 1
                            rights_of_batch += 1
                        else:
                            if (class_predicted,real_class) not in wrong_predictions_to_analize:
                                wrong_predictions_to_analize[int(class_predicted),int(real_class)] = 1
                            else:
                                wrong_predictions_to_analize[(class_predicted,real_class)] += 1
                        print('class_predicted:{}   ///   real_class:{}\n'.format(class_predicted,real_class))
                    except:pass
                print('samples corrected:{} / {}'.format(rights_of_batch,s_mini_batch))
                accuracy = 100 * right/samples
                print('total # of samples seen: {} /// total # of right predictions: {} /// Accuracy: {} %'.format(samples,right,accuracy))


        with open(f'{name_for_save}.csv', 'w') as f:
            for key in wrong_predictions_to_analize.keys():
                f.write("%s, %s\n" % (key, wrong_predictions_to_analize[key]))
            
        return accuracy
    
    
    test_loader = DataLoader(test_dataset, batch_size = s_mini_batch)
    validation_loader = DataLoader(validation_dataset, batch_size = s_mini_batch)
    
    model.eval()
    print('Evaluation Starting!')
    sleep(10)
    accuracy_test = evaluate_over(test_loader,'Error_of_test')
    print("EVALUATION OVER TEST SET DONE\n\nEVALUATION OVER VALIDATION SET STARTING")
    sleep(10)
    accuracy_validation = evaluate_over(validation_loader,'Error_of_validation')


    print('accuracy over test set : {}'.format(accuracy_test))
    print('accuracy over Validation set : {}'.format(accuracy_validation))