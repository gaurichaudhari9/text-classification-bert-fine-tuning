# Text Classification with BERT Fine-Tuning

## Project Overview

This project involves creating a text classification model to distinguish between quotes from Star Wars and Friends TV shows. The model is implemented by fine-tuning a pretrained BERT model on a custom dataset.

The key steps are:

1. Create a dataset with quotes from Star Wars and Friends shows (100 quotes from each)
2. Preprocess data - clean quotes, add labels 
3. Split data into train (160) and test (40) sets
4. Tokenize and pad the text sequences
5. Fine-tune a pretrained BERT model on the training data 
6. Evaluate model performance on the test set

## Data Collection

The quotes dataset was webscraped using Selenium and parsed using BeautifulSoup. The Star Wars quotes were taken from [this website](https://www.rd.com/article/star-wars-quotes/) and Friends quotes from [this website](https://www.telltalesonline.com/47670/friends-quotes/).

The final dataset contains 200 quotes - 100 labeled 0 for Star Wars and 100 labeled 1 for Friends.

## Model Implementation

The BERT base uncased model from HuggingFace Transformers is used. The model is first tokenized and then fine-tuned on the training data for 3 epochs.

The key aspects are:

- Tokenize quotes using BertTokenizer
- Add padding & truncation to a max length of 512
- Convert labels to numpy arrays
- Create TensorFlow datasets for train and test
- Compile model with Adam optimizer and Sparse Categorical Crossentropy loss
- Train for 3 epochs with a batch size of 64

## Results
Overall, the project demonstrates how BERT can be effectively fine-tuned on a small custom text classification dataset.
- The model was able to achieve **80%** accuracy on the test set, indicating BERT can capture distinguishing features between the two text styles even with a small dataset.
- However, 80% accuracy suggests there is still room for improvement.
- The relatively small size of the dataset (200 samples) is likely a limiting factor on model performance.
- Increasing to 5 epochs from 3 led to improved accuracy, suggesting the model could benefit from being trained for more epochs.
- Decreasing batch size from 64 to 32 also improved accuracy, indicating smaller batch sizes may work better for this dataset.
- The model is able to classify the quotes with moderate success. But generating entirely new text in the Star Wars or Friends style is likely to be more challenging and require more advanced techniques.


Some ways to further improve the model are:
- Use a larger dataset with more quotes per class
- Try different pretrained models like RoBERTa or DistilBERT
- Tune hyperparameters like batch size, learning rate and epochs
- Use additional regularization techniques like dropout



`Overall, the results demonstrate proof of concept for using BERT for this text classification task, but also highlight opportunities for further improving model accuracy and generalization ability.`
