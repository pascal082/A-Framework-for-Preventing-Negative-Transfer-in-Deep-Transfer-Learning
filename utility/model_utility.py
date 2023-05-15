import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score,f1_score
from transformers import BertModel, BertTokenizer
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np


num_epochs=20

#Custom BERT-based model with aleatoric and epistemic uncertainty
class CustomBertModel(nn.Module):
    def __init__(self, num_classes,bert_model):
        super(CustomBertModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)
        self.aleatoric = nn.Linear(bert_model.config.hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0, :]  # Use [CLS] token representation
        #pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        aleatoric_uncertainty = self.softplus(self.aleatoric(pooled_output))
        return logits, aleatoric_uncertainty

# Importance weighting
def compute_importance_weights(model,inputs,masks,train_labels):
    # Perform some calculations to compute importance weights based on input data


    # Evaluate the model to get prediction probabilities
    model.eval()
    with torch.no_grad():
        outputs = model(inputs, attention_mask=masks)
        logits = outputs[0]
        probabilities = torch.softmax(logits, dim=1)

    # Calculate the error rate or difficulty measure for each sample
    error_rates = 1 - probabilities[range(len(train_labels)), train_labels]

    # Calculate the weights inversely proportional to the error rate
    weights = 1 / error_rates
    importance_weights = weights / torch.sum(weights)  # Normalize the weights

    # Print the assigned weights
    #print(importance_weights)
    return importance_weights


# Define the loss function with uncertainty modeling
class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets, aleatoric_uncertainty, importance_weights):
        aleatoric_loss = torch.mean(aleatoric_uncertainty)
        epistemic_loss = self.loss_fn(logits, targets) * importance_weights
        total_loss = aleatoric_loss + epistemic_loss
        return total_loss.mean()

# Define the loss function
class LossImportantWeight(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets,importance_weights):

        loss = self.loss_fn(logits, targets)* importance_weights

        return loss.mean()

# Define the loss function
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):

        loss = self.loss_fn(logits, targets)

        return loss.mean()

#get clmmse score:

def get_clmmse_score(target_predictions,target_labels,importance_weights,train_error_rate,test_error_rate):
    # Assuming you have the predicted values (pf), ground truth (pXq), and importance weights (iw)
    # stored in numpy arrays
    pf = target_predictions
    pXq = target_labels
    iw = importance_weights

    # Compute the covariance matrix Cov(pf, pXq) using importance weights
    covariance_matrix = np.cov(pf, pXq, aweights=iw)

    # Extract the element corresponding to the covariance between pf and pXq
    covariance_pfpXq = covariance_matrix[1, 1]

    # Compute the variance Var(pXq) using importance weights
    variance_pXq = np.var(pXq, ddof=0)

    # Compute the estimated variance ˆμβ
    estimated_variance = ((covariance_pfpXq / variance_pXq) * train_error_rate.numpy()) + test_error_rate.numpy()

    # Print the estimated variance
    print("Estimated Variance (ˆμβ):", estimated_variance)
    return  estimated_variance
#Get metrics
def get_metrics(target_labels,target_predictions):
    bacc = balanced_accuracy_score(target_labels, target_predictions)
    # Calculate F1 score
    f1 = f1_score(target_labels, target_predictions)
    # Print the balanced accuracy score
    print("Balanced Accuracy Score:", bacc)
    print("f1 Score:", f1)
    return bacc, f1

#Get NTD
def get_error_rate(target_labels,target_predictions):
    # Calculate the number of incorrect predictions
    num_errors = sum(target_labels[i] != target_predictions[i] for i in range(len(target_labels)))

    # Calculate the error rate
    error_rate = num_errors / len(target_labels)
    return error_rate


def encode_data_for_bert_model(tokenizer,train_texts,max_length):
    # Tokenize and encode the training and target data
    train_inputs = tokenizer.batch_encode_plus(
        train_texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        pad_to_max_length=True
    )
    train_input_ids = train_inputs['input_ids']
    train_attention_mask = train_inputs['attention_mask']
    return train_inputs,train_input_ids,train_attention_mask




def train_model(model,loss_f,optimizer,train_inputs,train_input_ids,train_attention_mask,train_labels,use_uncertainty,use_importance_weights):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        logits, aleatoric_uncertainty = model(input_ids=train_input_ids, attention_mask=train_attention_mask)
        importance_weights = compute_importance_weights(model, train_inputs['input_ids'], train_attention_mask,
                                                        train_labels)

        if use_uncertainty and use_importance_weights:
            loss = loss_f(logits, train_labels, aleatoric_uncertainty, importance_weights)
        elif use_importance_weights and not use_uncertainty:
            loss = loss_f(logits, train_labels, importance_weights)
        else:
            loss = loss_f(logits, train_labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step
        optimizer.step
    return model, importance_weights




# Tokenize the input sequences using BERT tokenizer
def tokenize_inputs(tokenizer, inputs, max_length):
    input_ids = []
    attention_masks = []
    for input_text in inputs:
        encoded_input = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_input['input_ids'])
        attention_masks.append(encoded_input['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

