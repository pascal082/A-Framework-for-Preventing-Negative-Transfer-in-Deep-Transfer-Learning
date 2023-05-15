from utility.model_utility import UncertaintyLoss, train_model, CustomBertModel, compute_importance_weights, \
    encode_data_for_bert_model, Loss, LossImportantWeight, get_error_rate, get_metrics, get_clmmse_score, \
    apply_domain_divergence_using_psi, tokenize_inputs
from utility.nlp_utility import process_csv, preprocess_text
from utility.software_utils import MatchMetric
import torch
from transformers import BertTokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions import Normal
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer,DistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pandas as pd
import copy


num_classes=2
default_lr=1e-4



def run(amazon_csv,domain_group_list,save_path = 'amazon_source_model',apply_psi= True, use_importance_weights=True, use_uncertainty=True, clmmse=True):
    """

    :param amazon_csv:
    :param domain_group_list:
    :return: source data and target data
    """

    df = pd.read_json(amazon_csv)
    output = df.drop_duplicates()
    output.groupby('domain').size()

    domain_group = domain_group_list #["beauty", "outdoor_living", "jewelry_&_watches"]
    #group_2_domains = ["cell_phones_&_service", "software", "office_products"]

    rem_elem = ""  # remove elemn from the list as we loop
    _bacc=[]
    _f1=[]
    _ntd=[]
    for element in range(len(domain_group)):
        new_group_domains = copy.deepcopy(domain_group)


        new_group_domains.remove(domain_group[element]) #

        source_data = df[df["domain"].isin(new_group_domains)]

        target_data = df[df["domain"] == domain_group[element]]


        #change label data to binary
        source_data['Sentiment'] = source_data['label'].map({'positive': 1, 'negative': 0})
        target_data['Sentiment'] = target_data['label'].map({'positive': 1, 'negative': 0})

        rem_elem = domain_group[element]
        new_group_domains.append(rem_elem)  #

        #remove data that might cause domain divergence
        if apply_psi:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            source_model = BertModel.from_pretrained('bert-base-uncased')
            source_data,target_data = apply_domain_divergence_using_psi(source_model,tokenizer,source_data,target_data)

        #pre-process data
        pre_process_source_data = " ".join(preprocess_text(source_data["text"]))
        source_data["text"] = pre_process_source_data

        pre_process_target_data = " ".join(preprocess_text(target_data["text"]))
        target_data["text"] = pre_process_target_data

        #get model
        model, tokenizer,lr = get_model(num_classes,source_data,target_data,clmmse=True)

        # Split the target dataset into training and testing subsets
        target_train_inputs, target_test_inputs = train_test_split(target_data, test_size=0.3, random_state=42)

        train_texts = source_data["text"].tolist()
        train_labels = source_data["Sentiment"].tolist()



        target_train_texts = target_train_inputs["text"].tolist()
        target_train_labels = target_train_inputs["Sentiment"].tolist()

        target_test_texts = target_test_inputs["text"].tolist()
        target_test_labels = target_test_inputs["Sentiment"].tolist()

        #
        target_texts = target_data["text"].tolist()

        max_length = max(len(tokenizer.tokenize(text)) for text in train_texts + target_texts)

        train_inputs,train_input_ids,train_attention_mask = encode_data_for_bert_model(tokenizer, train_texts, max_length)
        target_train_inputs,target_train_input_ids,target_train_attention_mask = encode_data_for_bert_model(tokenizer, target_train_texts, max_length)
        target_test_inputs, target_test_input_ids, target_test_attention_mask = encode_data_for_bert_model(tokenizer,
                                                                                                              target_test_texts,
                                                                                                              max_length)

        train_labels = torch.tensor(train_labels)

        target_train_labels = torch.tensor(target_train_labels)
        target_test_labels = torch.tensor(target_test_labels)


        # Initialize the loss function

        if use_uncertainty and use_importance_weights:
           loss_fn = UncertaintyLoss()
        elif use_importance_weights and not use_uncertainty:
            loss_fn = LossImportantWeight()
        else:
            loss_fn = Loss()


        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Initialize the uncertainty loss function
        loss_f = loss_fn()
        bacc_list=[]
        f1_list=[]
        ntd_list=[]

        for i in range(10):

            #train model
            model,importance_weights = train_model(model, loss_f, optimizer, train_inputs, train_input_ids, train_attention_mask, train_labels,
                        use_uncertainty, use_importance_weights)

            #save model
            save_path= save_path + "_" + i +"_.pth"
            torch.save(model.state_dict(), save_path)

            # Make predictions
            with torch.no_grad():
                outputs = model(target_test_input_ids, attention_mask=target_test_attention_mask)
                logits = outputs[0]
                probabilities = torch.softmax(logits, dim=1)
                test_predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

            #calulete target error rate
            target_error_rate = get_error_rate(target_test_labels, test_predictions)



            #finetune with target data
            finetuned_model = model(num_classes)
            finetuned_model.load_state_dict(torch.load(save_path))

            # train model
            finetuned_model,importance_weights = train_model(finetuned_model, loss_f, optimizer, target_train_inputs, target_train_input_ids, target_train_attention_mask,
                                          target_train_labels,use_uncertainty, use_importance_weights)

            #make target prediction
            with torch.no_grad():
                outputs = finetuned_model(target_test_input_ids, attention_mask=target_test_attention_mask)
                logits = outputs[0]
                probabilities = torch.softmax(logits, dim=1)
                target_test_predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

            target_source_error_rate = get_error_rate(target_test_labels, target_test_predictions)
            ntd = target_source_error_rate - target_error_rate  #get negative transfer degree
            bacc, f1 = get_metrics(target_test_labels, target_test_predictions)
            bacc_list.append(bacc)
            f1_list.append(f1)
            ntd_list.append(ntd)

        #average the predition
        bacc_= sum(bacc_list) / len(bacc_list)
        f1_ = sum(f1_list) / len(f1_list)
        ntd_= sum(ntd_list) / len(ntd_list)

    #get metrics for all run
    _bacc.append(bacc_)
    _f1.append(bacc_)
    _ntd.append(bacc_)

    return _bacc, _f1, _ntd





def get_model(num_classes,source_data,target_data,clmmse=False):
    # Initialize BERT model and tokenizer
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    if clmmse:
        lr_list = [1e-3,1e-4,1e-5,1e-6,1e-7]
        clmmse_value = []
        loss_fn = LossImportantWeight()

        # Split the source dataset into training and testing subsets
        source_train_inputs, source_test_inputs = train_test_split(source_data, test_size=0.3, random_state=42)

        # prepare data
        train_texts = source_train_inputs["text"].tolist()
        train_labels = source_train_inputs["Sentiment"].tolist()

        train_test_texts = source_test_inputs["text"].tolist()
        train_test_labels = source_test_inputs["Sentiment"].tolist()

        target_texts = target_data["text"].tolist()
        target_labels = target_data["Sentiment"].tolist()

        max_length = max(len(tokenizer.tokenize(text)) for text in train_texts + target_texts)

        train_inputs, train_input_ids, train_attention_mask = encode_data_for_bert_model(tokenizer, train_texts,max_length)
        train_test_inputs, train_test_input_ids, train_test_attention_mask = encode_data_for_bert_model(tokenizer, train_test_texts,
                                                                                         max_length)
        target_inputs, target_input_ids, target_attention_mask = encode_data_for_bert_model(tokenizer,target_texts,max_length)


        train_labels = torch.tensor(train_labels)
        train_test_labels = torch.tensor(train_test_labels)
        target_labels = torch.tensor(target_labels)


        #get the best source model using clmmse
        clmmse_score=[]
        for i in range(len(lr_list)):
            _clmmse_score =[]
            for j in range(10):
                optimizer = torch.optim.Adam(bert_model.parameters(), lr=lr_list[i])


                #train model
                model,importance_weights = train_model(bert_model, loss_fn, optimizer, train_inputs, train_input_ids, train_attention_mask, train_labels,
                            use_uncertainty=False, use_importance_weights=True)

                # save model
                save_path = 'clmmse_amazon_source_model_' + i +"_.pth"
                torch.save(model.state_dict(), save_path)

                # Make predictions on source test set
                with torch.no_grad():
                    outputs = model(train_test_input_ids, attention_mask=train_test_attention_mask)
                    logits = outputs[0]
                    probabilities = torch.softmax(logits, dim=1)
                    test_predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

                # calculate source error rate
                train_error_rate = get_error_rate(train_test_labels, test_predictions)

                # Make predictions on target set
                with torch.no_grad():
                    outputs = model(target_input_ids, attention_mask=target_attention_mask)
                    logits = outputs[0]
                    probabilities = torch.softmax(logits, dim=1)
                    test_predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

                # calculate target error rate
                target_error_rate = get_error_rate(target_labels, test_predictions)
                score =train_error_rate -target_error_rate

                #get clmmse score
                score = get_clmmse_score(test_predictions, target_labels, importance_weights, train_error_rate,
                                 target_error_rate)


                _clmmse_score.append(score)
            average_clmmse_score = sum(_clmmse_score) / len(_clmmse_score)

            clmmse_score.append(average_clmmse_score)

            #get the max value
            max_value = max(clmmse_score)
            max_value = clmmse_score.index(max_value)
            model =  model(num_classes)

            save_path = 'clmmse_amazon_source_model_' + max_value + "_.pth"
            model.load_state_dict(torch.load(save_path))

            return model, tokenizer, lr_list[max_value]

    else:
        #use the dault lr value and return
        lr = default_lr
        # Initialize the custom model
        model = CustomBertModel(num_classes=num_classes, bert_model=bert_model)
        return model, tokenizer, lr


def apply_domain_divergence_using_psi(source_model,tokenizer,source_data,target_data):
    """

    :param source_data:
    :param target_data:
    :return: source data and the filtered target data
    """



    # Aggregate PSI scores row by row
    psi_scores = []
    filtered_target_data = []
    for _, (source_row, target_row) in enumerate(zip(source_data.iterrows(), target_data.iterrows())):
        _, source_data_row = source_row
        _, target_data_row = target_row
        # print(target_data_row)
        psi_score = calculate_PSI(source_model, source_data_row.to_frame().T, target_data_row.to_frame().T, tokenizer)
        if psi_score <= 0.05:
            filtered_target_data.append(target_data_row.to_frame().T)
        psi_scores.append(psi_score)

    # Aggregate PSI scores
    aggregate_psi = sum(psi_scores) / len(psi_scores)
    print("Aggregate PSI score:", aggregate_psi)
    return source_data, filtered_target_data

# Calculate the PSI between two DataFrames
def calculate_PSI(source_model, source_data, target_data, tokenizer):
    source_texts = source_data["text"].tolist()
    target_texts = target_data["text"].tolist()

    max_length = max(len(tokenizer.tokenize(text)) for text in source_texts + target_texts)

    source_input_ids, source_attention_masks = tokenize_inputs(tokenizer, source_texts, max_length)
    target_input_ids, target_attention_masks = tokenize_inputs(tokenizer, target_texts, max_length)

    # Get BERT embeddings for the source and target data
    with torch.no_grad():
        source_embeddings = source_model(source_input_ids, attention_mask=source_attention_masks)[0].mean(dim=1)
        target_embeddings = source_model(target_input_ids, attention_mask=target_attention_masks)[0].mean(dim=1)

    # Compute the probability scores for the source and target data
    source_probs = torch.softmax(source_embeddings, dim=1)
    target_probs = torch.softmax(target_embeddings, dim=1)

    # Calculate the PSI
    psi = torch.sum((target_probs - source_probs) * torch.log(target_probs / source_probs))
    return psi.item()






