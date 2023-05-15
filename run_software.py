from test import tokenize_inputs
from utility.model_utility import LossImportantWeight, encode_data_for_bert_model, train_model, get_error_rate, \
    get_clmmse_score, CustomBertModel, UncertaintyLoss, Loss, get_metrics
from utility.software_utils import MatchMetric
import torch
from transformers import BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
import pandas as pd
import pandas as pd
import copy

num_classes=2
default_lr=1e-4

def run():
    Xsource, Ysource, Xtarget, Ytarget, lenSource, loc = MatchMetric(
        "Data/SoftwareData/AEEEM/EQ.arff", "Data/SoftwareData/AEEEM/JDT.arff", split=True, merge=True)

#
def run(software_group_list, save_path='software_source_model', apply_psi=True, use_importance_weights=True,
        use_uncertainty=True, clmmse=True):
    """
    :param domain_group_list:
    :return: source data and target data
    """

    #software_group_list =["Data/SoftwareData/AEEEM/JDT.arff","Data/SoftwareData/AEEEM/EQ.arff","Data/SoftwareData/AEEEM/PDE.arff"]
    software_group = software_group_list

    rem_elem = ""  # remove elemn from the list as we loop
    _bacc = []
    _f1 = []
    _ntd = []
    for element in range(len(software_group)):
        new_source_group = copy.deepcopy(software_group)

        new_source_group.remove(software_group[element])  #remove first element

        Xsource, Ysource, Xtarget, Ytarget, lenSource, loc = MatchMetric(
            new_source_group, software_group[element] , split=True, merge=True)

        rem_elem = software_group_list[element]
        new_source_group.append(rem_elem)  #

        # remove data that might cause domain divergence
        if apply_psi:
            model_name = "microsoft/codebert-PT"
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            source_model = RobertaModel.from_pretrained(model_name)

            source_data, target_data = apply_domain_divergence_using_psi(source_model, tokenizer, Xsource,
                                                                         Xtarget)


        # get model
        model, tokenizer, lr = get_model(num_classes, source_data, target_data,Ysource,Ytarget, clmmse=True)

        # Split the target dataset into training and testing subsets
        target_train_inputs, target_test_inputs, target_train_label, target_test_label = train_test_split(target_data, Ytarget,test_size=0.3, random_state=42)

        train_texts = source_data.tolist()
        train_labels = Ysource.tolist()

        target_train_texts = target_train_inputs.tolist()
        target_train_labels = target_train_label.tolist()

        target_test_texts = target_test_inputs.tolist()
        target_test_labels =target_test_label.tolist()

        #
        target_texts = Ytarget.tolist()

        max_length = max(len(tokenizer.tokenize(text)) for text in train_texts + target_texts)

        train_inputs, train_input_ids, train_attention_mask = encode_data_for_bert_model(tokenizer, train_texts,
                                                                                         max_length)
        target_train_inputs, target_train_input_ids, target_train_attention_mask = encode_data_for_bert_model(
            tokenizer, target_train_texts, max_length)
        target_test_inputs, target_test_input_ids, target_test_attention_mask = encode_data_for_bert_model(
            tokenizer,
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
        bacc_list = []
        f1_list = []
        ntd_list = []

        for i in range(10):
            # train model
            model, importance_weights = train_model(model, loss_f, optimizer, train_inputs, train_input_ids,
                                                    train_attention_mask, train_labels,
                                                    use_uncertainty, use_importance_weights)

            # save model
            save_path = save_path + "_" + i + "_.pth"
            torch.save(model.state_dict(), save_path)

            # Make predictions
            with torch.no_grad():
                outputs = model(target_test_input_ids, attention_mask=target_test_attention_mask)
                logits = outputs[0]
                probabilities = torch.softmax(logits, dim=1)
                test_predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

            # calculate target error rate
            target_error_rate = get_error_rate(target_test_labels, test_predictions)

            # finetune with target data
            finetuned_model = model(num_classes)
            finetuned_model.load_state_dict(torch.load(save_path))

            # train model
            finetuned_model, importance_weights = train_model(finetuned_model, loss_f, optimizer,
                                                              target_train_inputs, target_train_input_ids,
                                                              target_train_attention_mask,
                                                              target_train_labels, use_uncertainty,
                                                              use_importance_weights)

            # make target prediction
            with torch.no_grad():
                outputs = finetuned_model(target_test_input_ids, attention_mask=target_test_attention_mask)
                logits = outputs[0]
                probabilities = torch.softmax(logits, dim=1)
                target_test_predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

            target_source_error_rate = get_error_rate(target_test_labels, target_test_predictions)
            ntd = target_source_error_rate - target_error_rate  # get negative transfer degree
            bacc, f1 = get_metrics(target_test_labels, target_test_predictions)
            bacc_list.append(bacc)
            f1_list.append(f1)
            ntd_list.append(ntd)

        # average the predition
        bacc_ = sum(bacc_list) / len(bacc_list)
        f1_ = sum(f1_list) / len(f1_list)
        ntd_ = sum(ntd_list) / len(ntd_list)

    # get metrics for all run
    _bacc.append(bacc_)
    _f1.append(bacc_)
    _ntd.append(bacc_)

    return _bacc, _f1, _ntd



def get_model(num_classes,source_data,target_data,Ysource,Ytarget,clmmse=False):
    # Initialize BERT model and tokenizer

    model_name = "microsoft/codebert-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    bert_model = RobertaModel.from_pretrained(model_name)

    if clmmse:
        lr_list = [1e-3,1e-4,1e-5,1e-6,1e-7]
        clmmse_value = []
        loss_fn = LossImportantWeight()

        # Split the source dataset into training and testing subsets
        source_train_inputs, source_test_inputs,train_labels,train_test_labels = train_test_split(source_data,Ysource, test_size=0.3, random_state=42)

        # prepare data
        train_texts = source_train_inputs.tolist()
        train_labels = train_labels.tolist()

        train_test_texts = source_test_inputs.tolist()
        train_test_labels = train_test_labels.tolist()

        target_texts = target_data.tolist()
        target_labels = Ytarget.tolist()

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
                save_path = 'clmmse_software_source_model_' + i +"_.pth"
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

            save_path = 'clmmse_software_source_model_' + max_value + "_.pth"
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
    source_texts = source_data[1].tolist()
    target_texts = target_data[1].tolist()

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


