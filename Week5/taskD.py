from transformers import BertTokenizer, BertModel
import pickle
import json
from tqdm import tqdm

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

with open('dataset/dataset.json','rb') as op:
    captions = json.load(op)

train_text_embs = []
test_text_embs = []

print('Generating embeddings...')
for  image_captions in (captions['images']):
    text_features = []
    for sentence in image_captions['sentences']:
        inputs = bert_tokenizer(sentence['raw'], return_tensors="pt")
        last_hidden = bert_model(**inputs)['last_hidden_state'].data.cpu().numpy()
        text_features.append(last_hidden[0])
    if image_captions['split'] == 'test':
        test_text_embs.append(text_features)
    else:
        train_text_embs.append(text_features)
        

with open('dataset/bertTrain_text_embs.pkl', 'wb') as op:
    pickle.dump(train_text_embs, op)

with open('dataset/bertTest_text_embs.pkl', 'wb') as op:
    pickle.dump(test_text_embs, op)


