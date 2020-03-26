#!/usr/bin/python
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import subprocess,sys
import pickle
import numpy as np
import re

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'jsonlines'])
import jsonlines

import nltk
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
pst=PorterStemmer()
Labels = ['contradiction','neutral','entailment','-']
s1_test,s2_test,label_test=[],[],[]
no_entry=[]

def text_clean(text):
  text=text.lower()
  texty=word_tokenize(text)
  texty=[word for word in texty if not word in stop_words]
  for i in range(len(texty)):
    texty[i]=pst.stem(texty[i])
  sen = (" ").join(texty)
  sen=re.sub('[^a-zA-Z]',' ', sen)
  sen = re.sub(r"\s+[a-zA-Z]\s+",' ', sen)
  sen = re.sub(r'\s+', ' ', sen)
  #stop_tokens=[word for words in ]
  return str(sen)


def test_data():
    global s1_test,s2_test,label_test
    str_test='data/snli_1.0_test.jsonl'
    co=0
    with jsonlines.open(str_test) as td:
        for line in td.iter():
            s1_test.append(text_clean(line['sentence1']))
            s2_test.append(text_clean(line['sentence2']))
            if(line['gold_label']=='-'):
                no_entry.append(co)
            label_test.append(Labels.index(line['gold_label'].lower()))
    label_test=np.array(label_test)
    print("{} sentences do not belong to any class.".format(len(no_entry)))

def logistic():
    filename='models/logistic.sav'
    sav_model=pickle.load(open(filename,'rb'))
    xtext_test=[]
    for i in range(len(s1_test)):
        fet=[]
        s1i,s2i=word_tokenize(s1_test[i]),word_tokenize(s2_test[i])
        for tokens in s1i:
            fet.append("s1_"+tokens)
        for tokens in s2i:
            fet.append("s2_"+tokens)
        xtext_test.append(" ".join(fet))
    print("Length of test set {}".format(len(xtext_test)))

    tfidf_mod=pickle.load(open('models/tfidf.pickle','rb'))
    feat_test=tfidf_mod.transform(xtext_test)
    print("TF-IDF feature set  : {}".format(feat_test.shape))

    pred_test = sav_model.predict(feat_test)
    fp=open("tfidf.txt","w")
    for line in pred_test:
        fp.write(Labels[line]+"\n")
    fp.close()
    print("\nTest accuracy : {}".format(sav_model.score(feat_test,label_test)))
    print("\n\n")

def lstm_model():
    model_test=load_model('models/lstm_312_keep.h5')
    tokens=pickle.load(open('models/tokens_180.lstm','rb'))

    max_seq=35
    num_class=len(Labels)
    seq1_test=tokens.texts_to_sequences(s1_test)
    seq2_test=tokens.texts_to_sequences(s2_test)
    xs1_test=pad_sequences(seq1_test,maxlen=max_seq)
    xs2_test=pad_sequences(seq2_test,maxlen=max_seq)

    ytest_arr=to_categorical(label_test,num_class)

    perf = model_test.evaluate([xs1_test, xs2_test], ytest_arr, verbose=1)
    yout=model_test.predict([xs1_test, xs2_test])
    yout=np.argmax(yout,axis=1)

    fp=open("deep_model.txt","w")
    for line in yout:
        fp.write(Labels[line]+"\n")
    fp.close()

    print("\nTest Accuracy:", perf[1])


def main():
    print('\nLoading data ..')
    test_data()

    print('\n\nRunning Logistic Regression ..\n')
    logistic()

    print('\n\nRunning LSTM Model ..\n')
    lstm_model()

main()
