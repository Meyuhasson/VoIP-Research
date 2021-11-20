#https://reposhub.com/python/deep-learning/rolczynski-Automatic-Speech-Recognition.html

from tqdm import tqdm
import pandas as pd
import automatic_speech_recognition as asr

DATA_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\extracted_datasets.csv"
OUTPUT_PROCESSED_FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\extracted_datasets_with_words_count.csv"

pipeline = asr.load('deepspeech2', lang='en')
pipeline.model.summary()

data = pd.read_csv(DATA_PATH).drop([r"Unnamed: 0"], axis=1)
data["words_in_conversation"] = data.apply(lambda row: pipeline.predict([asr.utils.read_audio(row["Audio_File_Path"])]), axis=1)
data.to_csv(OUTPUT_PROCESSED_FEATURES_PATH, index=False)

#file = FILE_PATH
#sample = asr.utils.read_audio(file)
#pipeline = asr.load('deepspeech2', lang='en')
#pipeline.model.summary()
#sentences = pipeline.predict([sample])
#print(sentences)