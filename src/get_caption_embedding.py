from flair.embeddings import FlairEmbeddings
from flair.data import Sentence
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import pandas as pd
import re
import numpy as np
from utils import parse_configs
import pickle
import os

config = parse_configs()

# uncomment below to download
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_str(string):
    string = re.sub(r'[^\w\s]', '', string)
    return string.strip().lower()

def preprocess_captions(data_df):
    captions = data_df['Caption']
    original_sentences = [_ for _ in captions]
    tokenize_sentences = []
    word_list_dict = {}
    for sentence in original_sentences:
        tmp = clean_str(sentence)
        word_list_tmp = tmp.split()
        doc_words = []
        for word in word_list_tmp:
            if word not in stop_words:
                doc_words.append(word)
                word_list_dict[word] = 1
        tokenize_sentences.append(doc_words)
    word_list = list(word_list_dict.keys())
    vocab_length = len(word_list)
    return tokenize_sentences

train_df = pd.read_csv(os.path.join(config.label_root, "train.csv"))
test_df = pd.read_csv(os.path.join(config.label_root, "test.csv"))

train_tokenized = preprocess_captions(train_df)
test_tokenized = preprocess_captions(test_df)


from flair.embeddings import StackedEmbeddings,CharacterEmbeddings,TransformerWordEmbeddings,WordEmbeddings
flair_embedding = StackedEmbeddings(
    [
        # character-level features
        CharacterEmbeddings(),
        WordEmbeddings('glove')
    ]
)

def get_flair_emb(token_seq):
    embedding_all = []
    for text in tqdm(token_seq):
        now_emb = []
        now_sent = Sentence(" ".join(text))
        flair_embedding.embed(now_sent)
        for w in now_sent:
            #print(w.embedding.shape, '??')
            now_emb.append(np.array(w.embedding.detach().cpu()))
        embedding_all.append(now_emb)
    return embedding_all


train_input_embs = get_flair_emb(train_tokenized)
test_input_embs = get_flair_emb(test_tokenized)

#train_input_embs = np.concatenate(train_input_embs, axis=0)
#test_input_embs = np.concatenate(test_input_embs, axis=0)

#np.save('/media/administrator/1305D8BDB8D46DEE/5329/cap_train_embed.npy', train_input_embs)
#np.save('/media/administrator/1305D8BDB8D46DEE/5329/cap_test_embed.npy', test_input_embs)

#print(train_input_embs.shape, test_input_embs.shape)


# Save caption emb
# train_caption_emb_file = "train_caption_emb.txt"
# test_caption_emb_file = "test_caption_emb.txt"
with open(config.cap_root, 'wb') as text:
    pickle.dump(train_input_embs, text)
with open(config.test_cap_root, 'wb') as text:
    pickle.dump(test_input_embs, text)