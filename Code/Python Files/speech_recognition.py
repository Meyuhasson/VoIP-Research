#https://reposhub.com/python/deep-learning/rolczynski-Automatic-Speech-Recognition.html

import automatic_speech_recognition as asr

file = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\external & internal datasets - after proccesing and tagging\malicious recordings\RTP Flood - external records\RTP_A_Host_Attacker\rtp.0.0_.wav"
sample = asr.utils.read_audio(file)
pipeline = asr.load('deepspeech2', lang='en')
pipeline.model.summary()
sentences = pipeline.predict([sample])
print(sentences)