import ast
import nltk
import optuna
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

# Data Preparation
nltk.download('punkt')  # Download the Punkt tokenizer


"""
BERT is tested for component identification, but the performance is not ideal. 
Majorly because of the bias of the dataset i.e. too much authors and titles.
"""


x = []
labels = []
with open("../../component_identification/train.txt", "r") as file:
    for line in file:
        instance = ast.literal_eval(line.strip())
        temp = []
        tokens = []
        for item in instance:
            if item[0] == '':
                continue
            each_tkn = nltk.word_tokenize(item[0])
            tokens.extend(each_tkn)

            for each in each_tkn:
                temp.append(item[1])
        x.append(tokens)
        labels.append(temp)







x_eval = []
labels_eval = []

with open("../../component_identification/eval.txt", "r") as file:
    for line in file:
        instance = ast.literal_eval(line.strip())
        temp = []
        tokens = []
        for item in instance:
            if item[0] == '':
                continue
            each_tkn = nltk.word_tokenize(item[0])
            tokens.extend(each_tkn)

            for each in each_tkn:
                temp.append(item[1])
        x_eval.append(tokens)
        labels_eval.append(temp)








# Define the label-to-id mapping
label2id = {"authors": 0, "title": 1, "volume": 2, "issue": 3, "pages": 4, "journal": 5, "year": 6, 'doi': 7, "X": 8}
id2label = {v: k for k, v in label2id.items()}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] + ["X"] * (n_subwords-1))

    return tokenized_sentence, labels

tokenized_texts_and_labels = [tokenize_and_preserve_labels(txt, lbl) for txt, lbl in zip(x, labels)]
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
label_lists = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

# Convert tokens to input IDs and labels to label IDs
input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts]
label_ids = [[label2id.get(l) for l in lab] for lab in label_lists]

# Padding & Truncation
MAX_LEN = 100
input_ids = [i + [0] * (MAX_LEN - len(i)) if len(i) < MAX_LEN else i[:MAX_LEN] for i in input_ids]
label_ids = [l + [label2id["X"]] * (MAX_LEN - len(l)) if len(l) < MAX_LEN else l[:MAX_LEN] for l in label_ids]

# Convert to tensors
input_ids = torch.tensor(input_ids)
label_ids = torch.tensor(label_ids)




tokenized_texts_and_labels_eval = [tokenize_and_preserve_labels(txt, lbl) for txt, lbl in zip(x_eval, labels_eval)]
tokenized_texts_eval = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels_eval]
label_lists_eval = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels_eval]

input_ids_eval = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_eval]
label_ids_eval = [[label2id.get(l) for l in lab] for lab in label_lists_eval]

input_ids_eval = [i + [0] * (MAX_LEN - len(i)) if len(i) < MAX_LEN else i[:MAX_LEN] for i in input_ids_eval]
label_ids_eval = [l + [label2id["X"]] * (MAX_LEN - len(l)) if len(l) < MAX_LEN else l[:MAX_LEN] for l in label_ids_eval]

input_ids_eval = torch.tensor(input_ids_eval)
label_ids_eval = torch.tensor(label_ids_eval)






# Custom Dataset
class NERDataset(Dataset):
    def __init__(self, input_ids, label_ids):
        self.input_ids = input_ids
        self.label_ids = label_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.label_ids[idx]}

dataset = NERDataset(input_ids, label_ids)
eval_dataset = NERDataset(input_ids_eval, label_ids_eval)



def objective(trial):

    # Define hyperparameters to be tuned
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 32, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.99)

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label2id))
    model.config.id2label = id2label
    model.config.label2id = label2id

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        weight_decay=weight_decay,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        use_mps_device=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    eval_results = trainer.evaluate()

    # Print the evaluation metrics
    print(f"\nTrial {trial.number} metrics:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")

    return eval_results["eval_loss"]


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Flatten the predictions and labels
    flat_predictions = predictions.flatten()
    flat_labels = labels.flatten()

    # Filter out padding tokens' labels
    flat_predictions = [pred_label for pred_label, true_label in zip(flat_predictions, flat_labels) if true_label != label2id["X"]]
    flat_labels = [label for label in flat_labels if label != label2id["X"]]

    results = {
        "precision": precision_score(flat_labels, flat_predictions, average='micro'),
        "recall": recall_score(flat_labels, flat_predictions, average='micro'),
        "f1": f1_score(flat_labels, flat_predictions, average='micro')
    }

    return results


def predict(input_text, model, tokenizer, MAX_LEN=100):
    # 1. Tokenize the input text
    tokens = tokenizer.tokenize(input_text)
    print(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(input_ids)
    # Padding & Truncation (assuming MAX_LEN as before)
    input_ids = input_ids + [0] * (MAX_LEN - len(input_ids)) if len(input_ids) < MAX_LEN else input_ids[:MAX_LEN]

    # Convert to tensor
    input_ids_tensor = torch.tensor([input_ids])  # Notice the additional [] to make it a batch of 1

    # 2. Make Predictions
    # Ensure model is in evaluation mode
    model.eval()

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids_tensor = input_ids_tensor.to(device)

    with torch.no_grad():
        logits = model(input_ids_tensor).logits

    # 3. Decode the logits to get the labels
    predicted_label_ids = torch.argmax(logits, dim=2).squeeze().tolist()  # Squeeze to remove batch dimension
    predicted_labels = [id2label[id] for id in predicted_label_ids]

    # Filter out padding labels
    original_tokens = tokens[:len(input_ids)]
    final_predictions = [label for token, label in zip(original_tokens, predicted_labels) if not token.startswith("##") and label != "X"]

    return final_predictions


def train_best_model(best_params):
    training_args = TrainingArguments(
        # ... your TrainingArguments parameters here ...
        output_dir='./results',
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["batch_size"],
        weight_decay=best_params["weight_decay"]
        # ... other arguments ...
    )

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label2id))
    model.to('cpu')
    model.config.id2label = id2label
    model.config.label2id = label2id

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return model



def predict_batch(input_texts, model, tokenizer, MAX_LEN=100):
    tokenized_data = [tokenizer.tokenize(text) for text in input_texts]
    input_ids_data = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_data]

    # Padding & Truncation
    input_ids_data = [ids + [0] * (MAX_LEN - len(ids)) if len(ids) < MAX_LEN else ids[:MAX_LEN] for ids in input_ids_data]

    # Convert to tensor
    input_ids_tensor = torch.tensor(input_ids_data)

    # Ensure model is in evaluation mode
    model.eval()

    # Move model and data to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids_tensor = input_ids_tensor.to(device)

    with torch.no_grad():
        logits = model(input_ids_tensor).logits

    # Decode the logits to get the labels
    predicted_label_ids = torch.argmax(logits, dim=2).tolist()

    all_predictions = []
    for tokens, label_ids in zip(tokenized_data, predicted_label_ids):
        predicted_labels = [id2label[id] for id in label_ids]
        # Filter out padding labels and subwords
        final_predictions = [label for token, label in zip(tokens, predicted_labels) if not token.startswith("##") and label != "X"]
        all_predictions.append(final_predictions)

    return all_predictions





if __name__ == '__main__':
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print(f"Best hyperparameters: {study.best_params}")
    best_params = study.best_params
    best_model = train_best_model(best_params)
    model_path = "/Users/jialong/PycharmProjects/RCMFS/test/transformers/model"
    best_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # After training, you can use the predict function like this:
    # x = "M. Eskandani, H. Derakhshankhah, R. Jahanban-Esfahlan, and M. Jaymand, \"Biomimetic alginate-based electroconductive nanofibrous scaffolds for bone tissue engineering application,\" Int J Biol Macromol, vol. 249, p. 125991, Jul 25 2023, doi: 10.1016/j.ijbiomac.2023.125991."
    # predicted_labels_for_x = predict(x, best_model, tokenizer)
    # print(predicted_labels_for_x)
    texts = [
        "SQUIRES, A., GERCHOW, L., MA, C., LIANG, E., TRACHTENBERG, M. & MINER, S. 2023. A multi-language qualitative study of limited English proficiency patient experiences in the United States. PEC Innov, 2, 100177.",
        "Prechelt L. Early stopping-but when?. InNeural Networks: Tricks of the trade 2002 Mar 28 (pp. 55-69). Berlin, Heidelberg: Springer Berlin Heidelberg.",
        "Prechelt, L. (1998). Automatic early stopping using cross validation: quantifying the criteria. Neural networks, 11(4), 761-767.",
        # ... any other texts ...
    ]
    predicted_labels_for_texts = predict_batch(texts, best_model, tokenizer)
    for text, labels in zip(texts, predicted_labels_for_texts):
        print(f"Text: {text}\nLabels: {labels}\n\n")

