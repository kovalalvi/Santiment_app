import spacy
from . import cnn_model_nlp
from torchtext import data
from torchtext import datasets
import torch

nlp = spacy.load('en')

model = cnn_model_nlp.initialize()
vocab_path = 'pytorch_model/vocab.pt'

def SentimentAnalyzer(sentence, model = model,  min_len = 5):
    device = 'cpu'
    model.eval()
    model.to(device)
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))

    #download vocabular
    vocab = torch.load(vocab_path)
    indexed = [vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))

    return prediction.item()

# print(predict_sentiment("This film is terrible"))
#
# print(predict_sentiment("This film is great"))
