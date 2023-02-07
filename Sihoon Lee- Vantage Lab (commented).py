#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary packages from PyTorch and Huggingface Transformers
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, AdamW
import os

# Disable annoying symlink warning from Huggingface Transformers
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Function that reads data and constructs a dataset
def create_dataset(src_file, ref_file):
    data = []
    
    # Read lines from files for source and reference sentences
    with open(src_file) as src_file, open(ref_file) as ref_file:
        for src, ref in zip(src_file, ref_file):
            # Strip newlines and split the sentence into individual words (tokens)
            src = src.rstrip()
            ref = ref.rstrip()
            src_tokens = src.split()
            ref_tokens = ref.split()
            
            # Compare each token in source and reference sentences
            for i, (src_token, ref_token) in enumerate(zip(src_tokens, ref_tokens)):
                # If tokens are different, append to data
                # src: source sentence
                # i: index of different token
                # ref_token: correct token from reference sentence
                if src_token != ref_token:
                    data.append((src, i, ref_token))
    
    # Returns list of tuples with source sentences and indexes of different tokens
    return data


class CorrectWordDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, mask_index, correct_word = self.data[index]

        # Tokenize the sentence
        inputs = self.tokenizer.encode_plus(
            sentence,
            max_length=128,     # Maximum length of the sentence
            truncation=True,    # Truncate if longer than max_length
            padding="max_length",  # Pad if shorter than max_length
            add_special_tokens=True,  # Add special tokens [CLS] and [SEP]
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'  # Return tensors
        )

        # Check whether the mask_index is outside of the max length
        if mask_index + 1 >= 128:
            return None

        # Squeeze redundant dimensions
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        mask_token_id = self.tokenizer.mask_token_id
        correct_token_id = self.tokenizer.encode(
            correct_word, add_special_tokens=False)[0]

        # Replace the token at the mask index with the mask token
        input_ids[mask_index + 1] = mask_token_id
        # Create a labels tensor with -100 everywhere other than the mask index
        labels = torch.full_like(input_ids, -100)
        labels[input_ids==mask_token_id] = correct_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def collate_fn(batch):
    # Remove None values from the batch
    batch = [item for item in batch if item is not None]
    
    # Pad the input_ids, attention_mask and labels to the same length
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def predict(model, tokenizer, dataloader, test_data, device):
    # Put the model in eval mode
    model.eval()
    index = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Get the output from the model
        output_dict = model(input_ids, attention_mask=attention_mask)
        # Get the predictions from the output logits
        predictions = output_dict.logits.argmax(dim=-1)
        
        # Get the original sentence, mask index, and correct word from the test data
        sentence, mask_index, _ = test_data[index]
        # Get the predicted word id
        predicted_word_id = predictions[0, mask_index + 1]
        # Convert the predicted word id to the word
        predicted_word = tokenizer.convert_ids_to_tokens([predicted_word_id])[0]

        # Print the original sentence and the proposed correct word
        print(f'Original sentence: {sentence.strip()}')
        print(f'Proposed correct word: {predicted_word}')
        
        index += 1


# In[11]:


def main():
    # Load the DistilBert tokenizer and model from the Hugging Face model hub
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    
    # Set the path for the source and reference files
    src_file = 'data/dev.src'
    ref_file = 'data/dev.ref0'
    # Create the dataset from the source and reference files
    data = create_dataset(src_file, ref_file)
    # Use a small subset of data
    data = data[:10]
    
    # Create the CorrectWordDataset object from the data and tokenizer
    dataset = CorrectWordDataset(data, tokenizer)
    # Create the data loader from the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Add this
    # Define the optimizer for the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-1)
    
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Transfer the model to the device
    model.to(device)

 # Reduce the number of epochs from 3 to 1
    num_epochs = 1
    for epoch in range(1):
        # Iterate over batches in the data loader
        for idx, batch in enumerate(dataloader):
            # Extract input_ids, attention_mask, and labels from the batch, send to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Zero the optimizer gradient
            optimizer.zero_grad()
            # Forward pass the inputs through the model
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            # Extract the loss from the output
            loss = output.loss
            # Compute backward gradients
            loss.backward()
            # Update optimizer step
            optimizer.step()
            
    # Load test data
    test_src_file = 'data/test/test.src'
    test_ref_file = 'data/test/test.ref0'
    test_data = create_dataset(test_src_file, test_ref_file)
    test_data = test_data[:10]
    test_dataset = CorrectWordDataset(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    predict(model, tokenizer, test_dataloader, test_data, device)

    # Save the trained model
    model.save_pretrained('jfleg-master/data')

if __name__ == '__main__':
    main()


# In[17]:


def main_pretrained():
    # Load pre-trained DistilBERT model and tokenizer
    model_path = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForMaskedLM.from_pretrained(model_path)

    # Load test data
    test_src_file = 'data/test/test.src'
    test_ref_file = 'data/test/test.ref0'
    test_data = create_dataset(test_src_file, test_ref_file)
    
    # Create DataLoader from test data
    test_dataset = CorrectWordDataset(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Send model to GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loop over test dataloader
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass of the model
        output_dict = model(input_ids, attention_mask=attention_mask)
        predictions = output_dict.logits.argmax(dim=-1)

        # Loop over the range of the length of the test dataset
        for index in range(len(test_data)):
            sentence, mask_index, _ = test_data[index]

            # Retrieve predicted token index
            predicted_word_id = predictions[index, mask_index + 1]
            
            # Decode the index to the corresponding word
            predicted_word = tokenizer.decode([predicted_word_id])

            # Print the original word, original sentence, and the proposed correct word
            print(f'Original word: {sentence.split()[mask_index]}')
            print(f'Original sentence: {sentence.strip()}')
            print(f'Proposed correct word: {predicted_word}')
            # This increment of 'index' is redundant because it is already incremented by the for loop
            index += 1
if __name__ == '__main__':
    main_pretrained()


# In[27]:


def predict_on_sentence(model, tokenizer, src_sentence, device):
    src_sentence = str(src_sentence)
    data = []
    src_words = src_sentence.split() # Split the source sentence into individual words
    
    # Use a for loop to create data, with index and words in src_words using enumerate
    for mask_index, src_word in enumerate(src_words):
        data.append((src_sentence, mask_index, None))
    
    # Create a dataset using CorrectWordDataset with the data and tokenizer
    dataset = CorrectWordDataset(data, tokenizer)
    # Create a dataloader using DataLoader with the created dataset and other parameters
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model.to(device) # Moves and/or casts the model parameters and buffers
    result = []
    index = 0
    for batch in dataloader: # For each batch in the dataloader
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        output_dict = model(input_ids, attention_mask=attention_mask) # Pass input_ids and attention_mask to model
        predictions = output_dict.logits.argmax(dim=-1) # Get the predicted words from the output_dict logits

        for _, mask_index, _ in data[index:index+len(predictions)]:
            predicted_word_id = predictions[index % 64, mask_index + 1] # predicted_word_id is equal to the predicted word in the predictions
            predicted_word = tokenizer.decode([predicted_word_id]) # Decode the predicted_word from predicted_word_id
            
            result.append(predicted_word.strip()) # Append the predicted word to the result list

            index += 1

    return ' '.join(result) # Return the result list as a single sentence

def main():
    # Load the pretrained model
    model_dir = "jfleg-master/data"  # replace with your own path
    model = DistilBertForMaskedLM.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    # Testing sentences
    sentences = [
        "There knowledge of those facts was incomplete!",
        "Their going to learn something new from the ML course."
    ]

    # Perform inference (i.e., making predictions) on the testing sentences
    predict_on_sentences(model, tokenizer, sentences)

if __name__ == "__main__":
    main()


# In[29]:


def calculate_accuracy(model, tokenizer, dataloader, test_data, device):
    model.eval() # Set the model to evaluation mode
    index = 0
    correct = 0 # Counter for the number of correct predictions

    for batch in dataloader: # For each batch in the dataloader
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        output_dict = model(input_ids, attention_mask=attention_mask) # Pass input_ids and attention_mask to model
        predictions = output_dict.logits.argmax(dim=-1) # Get the predicted words from the output_dict logits

        # For each item in the test_data within the batch
        for _, mask_index, ref_word in test_data[index:index+len(predictions)]:
            predicted_word_id = predictions[index % 64, mask_index + 1] # Get the predicted word id at the corresponding index
            predicted_word = tokenizer.decode([predicted_word_id]).strip() # Decode the predicted_word from predicted_word_id

            if predicted_word == ref_word: # If predicted_word matches ref_word
                correct += 1 # Increment the number of correct predictions

            index += 1 # Increment the index to move to next item in batch

    return correct / len(test_data) # Calculate accuracy as ratio of correct predictions to total number of test data


# In[30]:


test_src_file = 'data/test/test.src'
test_ref_file = 'data/test/test.ref0'
test_data = create_dataset(test_src_file, test_ref_file)
test_dataset = CorrectWordDataset(test_data, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

accuracy = calculate_accuracy(model, tokenizer, test_dataloader, test_data, device)
print(f'Accuracy: {accuracy}')

