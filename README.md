# Vantage-Labs-Grammar-Correction

# Context-Aware Misused Word Correction

## Overview
The goal of this project is to develop an algorithm that can detect and correct misused words in a given sentence using deep learning techniques. In this project, I used the PyTorch library and the DistilBert model for the task. The model will read a sentence and then return a sentence with the proposed correct word usage.

This project consists of two main parts: 1) Fine-tuning the DistilBert model using the JFLEG dataset, and 2) Predicting and evaluating the model's performance.

## Dataset
I used the JFLEG (JHU FLuency-Extended GUG) dataset, which consists of a large number of sentences with grammatical errors that were collected from the web. The dataset also contains references, which are created by human editors, to represent the corrections for each source sentence. 

I tokenized the source and reference sentences and then used them to fine-tune the DistilBert model. The trained model was then used to predict the correct word usage on new sentences.

## Model Training
The first part of the code contains the code to fine-tune the DistilBert model using the source and reference sentences from the JFLEG dataset. The model was fine-tuned to predict the correct word usage in a sentence in the context of the word's appearance. The trained model was then saved for further use.

## Model Evaluation
The second part contains the code to evaluate the trained model's performance on a test dataset from JFLEG. The test sentences were fed into the trained model, and the file contains a function to make predictions on new input sentences. Given a new sentence, the function returns a sentence with the proposed correct word usage.

## Requirements
- Python 3
- PyTorch
- transformers
- nltk
- JFLEG dataset (https://github.com/keisks/jfleg)
