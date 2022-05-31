import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

pd.set_option('display.max_columns', 4)
pd.set_option('display.max_colwidth', 16)

oov_tok = '<OOV_TOK>'
vocab_size = 100000
padding_type = 'post'
trunc_type = 'post'
max_length = 150
test_portion = 0.2
embedding_dim = 16

data_vocab = pd.read_csv('KAIAccess_final_ver3-numberremoved-nulltextremoved-fixed.csv')
data_vocab.review_text=data_vocab.review_text.astype(str)
data_vocab.drop(columns=['app_ver_name', 'reviewer_language','device', 'review_month', 'star_rating'],
                inplace = True)
data_predict = pd.read_csv('labelled_tambahan.csv')

sent_model = load_model('nonull_model_sent-nonumber.h5')
topic_model = load_model('nonull_model_topic-nonumber.h5')
detail_model = load_model('nonull_model_detail-nonumber.h5')

def process_vocab(data_vocab):
    train_set = data_vocab
    text_train = train_set['review_text']
    sent_train = train_set['sentiment']
    topic_train = train_set['topic']
    detail_train = train_set['detail_topic']

    tokenizer = Tokenizer(vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(text_train)
    text_word_index = tokenizer.word_index

    sent_tokenizer = Tokenizer()
    sent_tokenizer.fit_on_texts(sent_train)
    sent_word_index = sent_tokenizer.word_index

    topic_tokenizer = Tokenizer()
    topic_tokenizer.fit_on_texts(topic_train)
    topic_word_index = topic_tokenizer.word_index

    detail_tokenizer = Tokenizer()
    detail_tokenizer.fit_on_texts(detail_train)
    detail_word_index = detail_tokenizer.word_index

    print("setences dict:")
    print(text_word_index)
    print("sentiment dict:")
    print(sent_word_index)
    print("topic dict:")
    print(topic_word_index)
    print("detail dict:")
    print(detail_word_index)

    return tokenizer, text_word_index, sent_tokenizer, sent_word_index,\
           topic_tokenizer, topic_word_index, detail_tokenizer, detail_word_index

def preprocessing(data_predict):
    data_predict.drop(columns=['sentiment', 'topic', 'detail_topic'], inplace =True)
    print(data_predict.columns)
    print(len(data_predict))
    # Cleaning punctuation, space, emoji, capital letter
    # Remove punctuation and emojis
    data_predict['Review Text'] = data_predict['Review Text'].str.replace('[^\w\s]', '')
    # Lowering Case
    data_predict['Review Text'] = data_predict['Review Text'].str.lower()
    # remove URLs
    data_predict['Review Text'] = data_predict['Review Text'].replace(r'http\S+', '', regex=True)\
        .replace(r'www\S+', '', regex=True)
    # remove newlines
    data_predict['Review Text'] = data_predict['Review Text'].str.replace('\n', ' ')
    # replace two space to one
    data_predict['Review Text'] = data_predict['Review Text'].str.replace('\s\s+', ' ', regex=True)
    # remove leading space
    data_predict['Review Text'] = data_predict['Review Text'].replace('^ +| +$', '', regex=True)
    #remove number
    data_predict['Review Text'] = data_predict['Review Text'].str.replace('\d+', '')
    #remove single letter
    data_predict['Review Text'] = data_predict['Review Text'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
    print(data_predict[data_predict['Review Text'].isnull()])
    #remove data that have no review text
    data_predict.drop(data_predict[data_predict['Review Text'].isnull()].index, inplace= True )

    # WITH NO STOPWORDS
    """f = open("id-stopwords.txt", "r")
    list = f.read().splitlines()
    print(list)
    data_predict['Review Text'] = data_predict['Review Text'].apply(lambda x: ' '.join([word for word in x.split()
                                                                        if word not in (list)]))
    """
    # manipulate NaN values
    data_predict.loc[(data_predict['App Version Name'].isnull()), 'App Version Name'] = 'unknown'
    data_predict.loc[(data_predict.Device.isnull()), 'device'] = 'unknown'

    return data_predict

def predict_sentiment(data_predict):
    input = np.array(data_predict['Review Text'])
    prediction = sent_model.predict(np.array(pad_sequences(tokenizer.texts_to_sequences(input),
                                                           padding=padding_type, maxlen=max_length,
                                                           truncating=trunc_type)))
    list_result = []
    for row in range(len(prediction)):
        result = prediction[row].tolist().index(np.max(prediction[row]))
        list_result.append(result)
        print("prediction- " + str(row) + " : " + str(result) + " with " + str(
            100 * max(prediction[row].tolist() * 100)) + " percentage")
    pre_result = []
    for s in list_result:
        pre_result.append(sent_tokenizer.index_word[s])

    data_predict['Sentiment Prediction'] = np.array(pre_result)
    print(data_predict[['Review Text', 'Sentiment Prediction']])
    return data_predict

def predict_topic(data_predict):
    input_text = np.array(data_predict['Review Text'])
    input_text = pad_sequences(tokenizer.texts_to_sequences(input_text),
                                                           padding=padding_type, maxlen=max_length,
                                                           truncating=trunc_type)
    input_sent = np.array(data_predict['Sentiment Prediction'])
    input_sent = pad_sequences(sent_tokenizer.texts_to_sequences(input_sent), maxlen=1)
    input = np.concatenate([input_text, input_sent], axis=1)
    prediction = topic_model.predict(np.array(input))
    list_result = []
    for row in range(len(prediction)):
        result = prediction[row].tolist().index(np.max(prediction[row]))
        list_result.append(result)
        print("prediction- " + str(row) + " : " + str(result) + " with " + str(
            100 * max(prediction[row].tolist() * 100)) + " percentage")
    pre_result = []
    for s in list_result:
        pre_result.append(topic_tokenizer.index_word[s])

    data_predict['Topic Prediction'] = np.array(pre_result)
    print(data_predict[['Review Text', 'Sentiment Prediction', 'Topic Prediction']])
    return data_predict

def predict_detail(data_predict):
    input_text = np.array(data_predict['Review Text'])
    input_text = pad_sequences(tokenizer.texts_to_sequences(input_text),
                               padding=padding_type, maxlen=max_length,
                               truncating=trunc_type)
    input_sent = np.array(data_predict['Sentiment Prediction'])
    input_sent = pad_sequences(sent_tokenizer.texts_to_sequences(input_sent), maxlen=1,
                               padding=padding_type, truncating=trunc_type)
    input_topic = np.array(data_predict['Topic Prediction'])
    input_topic = pad_sequences(topic_tokenizer.texts_to_sequences(input_topic), maxlen=1,
                               padding=padding_type, truncating=trunc_type)
    input = np.concatenate([input_text, input_sent, input_topic], axis=1)
    prediction = detail_model.predict(np.array(input))
    list_result = []
    for row in range(len(prediction)):
        result = prediction[row].tolist().index(np.max(prediction[row]))
        list_result.append(result)
        print("prediction- " + str(row) + " : " + str(result) + " with " + str(
            100 * max(prediction[row].tolist() * 100)) + " percentage")

    preresult = []

    for aa in list_result:
        preresult.append(detail_tokenizer.index_word[aa])

    data_predict['Detail Prediction'] = np.array(preresult)
    print(data_predict[['Review Text', 'Sentiment Prediction', 'Topic Prediction', 'Detail Prediction']])
    return data_predict

def rename(data_predict):
    data_predict.loc[(data_predict['Sentiment Prediction'] == 'positif'), 'Sentiment Prediction'] = 'positive'
    data_predict.loc[(data_predict['Sentiment Prediction'] == 'negatif'), 'Sentiment Prediction'] = 'negative'
    data_predict.loc[(data_predict['Sentiment Prediction'] == 'netral'), 'Sentiment Prediction'] = 'neutral'

    data_predict.loc[(data_predict['Topic Prediction'] == 'pemesanan'), 'Topic Prediction'] = 'booking'
    data_predict.loc[(data_predict['Topic Prediction'] == 'pembayaran'), 'Topic Prediction'] = 'payment'
    data_predict.loc[(data_predict['Topic Prediction'] == 'pembatalan'), 'Topic Prediction'] = 'cancellation'
    data_predict.loc[(data_predict['Topic Prediction'] == 'pengaturan'), 'Topic Prediction'] = 'settings'
    data_predict.loc[(data_predict['Topic Prediction'] == 'registrasilogin'), 'Topic Prediction'] = 'registration-login'
    data_predict.loc[(data_predict['Topic Prediction'] == 'error'), 'Topic Prediction'] = 'errors'

    data_predict.loc[(data_predict['Detail Prediction'] == 'errorpemesanan'), 'Detail Prediction'] = 'booking error'
    data_predict.loc[(data_predict['Detail Prediction'] == 'maxpemesanan'), 'Detail Prediction'] = 'booking maximum'
    data_predict.loc[(data_predict['Detail Prediction'] == 'editpemesanan'), 'Detail Prediction'] = 'edit booking'
    data_predict.loc[(data_predict['Detail Prediction'] == 'bookbedaid'), 'Detail Prediction'] = 'different ID booking'
    data_predict.loc[(data_predict['Detail Prediction'] == 'getjadwal'), 'Detail Prediction'] = 'get schedule'
    data_predict.loc[(data_predict['Detail Prediction'] == 'errorpembayaran'), 'Detail Prediction'] = 'payment error'
    data_predict.loc[(data_predict['Detail Prediction'] == 'metodebayar'), 'Detail Prediction'] = 'payment method'
    data_predict.loc[(data_predict['Detail Prediction'] == 'errorpembatalan'), 'Detail Prediction'] = 'cancellation error'
    data_predict.loc[(data_predict['Detail Prediction'] == 'announcebatal'), 'Detail Prediction'] = 'cancellation notification'
    data_predict.loc[(data_predict['Detail Prediction'] == 'registrasi'), 'Detail Prediction'] = 'registration'
    data_predict.loc[(data_predict['Detail Prediction'] == 'editprofil'), 'Detail Prediction'] = 'edit profile'
    data_predict.loc[(data_predict['Detail Prediction'] == 'baik'), 'Detail Prediction'] = 'good'
    data_predict.loc[(data_predict['Detail Prediction'] == 'lain'), 'Detail Prediction'] = 'others'
    data_predict.loc[(data_predict['Detail Prediction'] == 'aksesapp'), 'Detail Prediction'] = 'app access'
    data_predict.drop(columns=['Unnamed: 0'], inplace=True)
    print(data_predict[['Review Text', 'Sentiment Prediction', 'Topic Prediction', 'Detail Prediction']])

if __name__ == '__main__':
    data_predict = preprocessing(data_predict)
    tokenizer, text_word_index, sent_tokenizer, sent_word_index, topic_tokenizer, topic_word_index, detail_tokenizer, detail_word_index = process_vocab(data_vocab)
    data_predict = predict_sentiment(data_predict)
    data_predict = predict_topic(data_predict)
    data_predict = predict_detail(data_predict)
    data_predict = rename(data_predict)
    #data_predict.to_csv('result_labelled_tambahan.csv')