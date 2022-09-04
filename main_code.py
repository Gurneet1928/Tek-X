from concurrent.futures import thread
import speech_recognition as sr
import moviepy.editor as mp
from pydub import AudioSegment
import threading
import collections
import text_summarise_a as tss
import nltk
import re
import string
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize as nlkt_sent_tokenize
from nltk.tokenize import word_tokenize as nlkt_word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd


# Global Dictionaries and Lists
dict_ans = {}
thread_res = []
final_text = ' '
answer = ' '
#Convert The Given Video to Audio
clip = mp.VideoFileClip(r"abcd.mp4")
clip.audio.write_audiofile(r"temp.wav")

#Use the SpeechRecognition Library to make chunks of the audio file and transcript them 
r = sr.Recognizer()
r.dynamic_energy_threshold = True
filename = "temp.wav"
audio = AudioSegment.from_wav(filename)

n = len(audio) #Legth of Audio file // *1000 to convert the milliseconds to seconds
i = 0
diff = 30*1000
st = 0*1000

def transcript(j): #Function to transcript specified {j} chunks of audio and store them in dictionary
    global dict_ans
    with sr.AudioFile("temp_chunk_{}.wav".format(j)) as source:
        audio_Data = r.record(source)
        r.adjust_for_ambient_noise(source, duration = 1)
        text = r.recognize_google(audio_Data)
        dict_ans[j]=text
        print(j," => ",dict_ans[j])

def similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score

def sent_tokenize(text):
    sents = nlkt_sent_tokenize(text)
    sents_filtered = []
    for s in sents:
        sents_filtered.append(s)
    return sents_filtered

def cleanup_sentences(text):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    sentences_cleaned = []
    for sent in sentences:
        words = nlkt_word_tokenize(sent)
        words = [w for w in words if w not in string.punctuation]
        words = [w for w in words if not w.lower() in stop_words]
        words = [w.lower() for w in words]
        sentences_cleaned.append(" ".join(words))
    return sentences_cleaned

def get_tf_idf(sentences):
    vectorizer = CountVectorizer()
    sent_word_matrix = vectorizer.fit_transform(sentences)

    transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
    tfidf = transformer.fit_transform(sent_word_matrix)
    tfidf = tfidf.toarray()

    centroid_vector = tfidf.sum(0)
    centroid_vector = np.divide(centroid_vector, centroid_vector.max())

    feature_names = vectorizer.get_feature_names()

    relevant_vector_indices = np.where(centroid_vector > 0.3)[0]

    word_list = list(np.array(feature_names)[relevant_vector_indices])
    return word_list

#Populate word vector with all embeddings.
#This word vector is a look up table that is used
#for getting the centroid and sentences embedding representation.
def word_vectors_cache(sentences, embedding_model):
    word_vectors = dict()
    for sent in sentences:
        words = nlkt_word_tokenize(sent)
        for w in words:
            word_vectors.update({w: embedding_model.wv[w]})
    return word_vectors

# Sentence embedding representation with sum of word vectors
def build_embedding_representation(words, word_vectors, embedding_model):
    embedding_representation = np.zeros(embedding_model.vector_size, dtype="float32")
    word_vectors_keys = set(word_vectors.keys())
    count = 0
    for w in words:
        if w in word_vectors_keys:
            embedding_representation = embedding_representation + word_vectors[w]
            count += 1
    if count != 0:
       embedding_representation = np.divide(embedding_representation, count)
    return embedding_representation


def summ_para(text, emdedding_model):
    raw_sentences = sent_tokenize(text)
    clean_sentences = cleanup_sentences(text)
    for i, s in enumerate(raw_sentences):
        print(i, s)
    for i, s in enumerate(clean_sentences):
        print(i, s)
    centroid_words = get_tf_idf(clean_sentences)
    print(len(centroid_words), centroid_words)
    word_vectors = word_vectors_cache(clean_sentences, emdedding_model)
    #Centroid embedding representation
    centroid_vector = build_embedding_representation(centroid_words, word_vectors, emdedding_model)
    sentences_scores = []
    for i in range(len(clean_sentences)):
        scores = []
        words = clean_sentences[i].split()

        #Sentence embedding representation
        sentence_vector = build_embedding_representation(words, word_vectors, emdedding_model)

        #Cosine similarity between sentence embedding and centroid embedding
        score = similarity(sentence_vector, centroid_vector)
        sentences_scores.append((i, raw_sentences[i], score, sentence_vector))
    sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)
    for s in sentence_scores_sort:
        print(s[0], s[1], s[2])
    count = 0
    sentences_summary = []
    #Handle redundancy
    for s in sentence_scores_sort:
        if count > 100:
            break
        include_flag = True
        for ps in sentences_summary:
            sim = similarity(s[3], ps[3])
            if sim > 0.95:
                include_flag = False
        if include_flag:
            sentences_summary.append(s)
            count += len(s[1].split())

        sentences_summary = sorted(sentences_summary, key=lambda el: el[0], reverse=False)

    summary = "\n".join([s[1] for s in sentences_summary])
    summ_output = open('text.txt', 'a+')
    summ_output.write(summary)

def sol_main(): #Big Function with lots of task
    global st,diff,i
    global dict_ans
    global thread_res
    global final_text
    while(st+diff<=n): #Makes chunks of audio, from start to chunk-1
        chunk = audio[st:st+diff]
        chunk.export("temp_chunk_{}.wav".format(i),format='wav')
        th = threading.Thread(target=transcript,args=(i,)) #Append the thread in a list
        thread_res.append(th)
        i = i+1
        st+=diff
    chunk = audio[st:] #Make the last chunk of audio if left
    chunk.export("temp_chunk_{}.wav".format(i),format='wav')
    th = threading.Thread(target=transcript,args=(i,)) #Append the thread in a list
    thread_res.append(th)
    for th in thread_res: #Execute all the threads
        th.start()
    for th in thread_res: #Wait for all the transcription to end
        th.join()
    od = collections.OrderedDict(sorted(dict_ans.items())) #Ordered dictionary to sort dictionary as per key values
    output = open("transcript.txt", "w")
    open('text.txt', 'w').close()
    for k, v in od.items():
        output.writelines(f'{v}\n')
    for k,v in od.items():
        text = str(v)
        clean_sentences = cleanup_sentences(text)
        words = []
        for sent in clean_sentences:
            words.append(nlkt_word_tokenize(sent))
        model = Word2Vec(words, min_count=1, sg = 1)
        summ_para(text, model)
        
sol_main()