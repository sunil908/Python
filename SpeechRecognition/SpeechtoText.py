# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:08:02 2018

@author: GEFA0728
"""

#from nltk import pos_tag
#from nltk import word_tokenize
#from nltk.corpus import stopwords

import speech_recognition as sr
# obtain path to "english.wav" in the same folder as this script
import argparse
import base64
import json

from googleapiclient import discovery
import httplib2
from oauth2client.client import GoogleCredentials


DISCOVERY_URL = ('https://{api}.googleapis.com/$discovery/rest?'
                 'version={apiVersion}')


def readaudio(audiofile):
    



def get_speech_service():
    credentials = GoogleCredentials.get_application_default().create_scoped(
        ['https://www.googleapis.com/auth/cloud-platform'])
    http = httplib2.Http()
    credentials.authorize(http)

    return discovery.build(
        'speech', 'v1beta1', http=http, discoveryServiceUrl=DISCOVERY_URL)

def main(speech_file):
    """Transcribe the given audio file.

    Args:
        speech_file: the name of the audio file.
    """
    with open(speech_file, 'rb') as speech:
        speech_content = base64.b64encode(speech.read())


    service = get_speech_service()
    service_request = service.speech().syncrecognize(
        body={
            'config': {
                'encoding': 'FLAC',  # raw 16-bit signed LE samples
                'sampleRate': 16000,  # 16 khz
                'languageCode': 'en-US',  # a BCP-47 language tag
            },
            'audio': {
                'content': speech_content.decode('UTF-8')
                }
            })
    response = service_request.execute()
    print(json.dumps(response))

if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument(
#        'speech_file', help='Full path of audio file to be recognized')
#    args = parser.parse_args()
    main('D:\Training Files\speech5.flac')






























#reftext="this is just reporting what shit in this age of what is best that is the two big issues what up the that will put to the death of tools spoil it"
#AUDIO_FILE = ("D:\Training Files\speech4.flac")
## AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "french.aiff")
## AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "chinese.flac")
# 
## use the audio file as the audio source
#r = sr.Recognizer()
#with sr.AudioFile(AUDIO_FILE) as source:
#    audio = r.record(source)  # read the entire audio file
#
#
## recognize speech using Sphinx
#try:
#    print("Google thinks you said " + r.recognize_google_cloud(audio))
##    print(wer(reftext, r.recognize_sphinx(audio)))
#except sr.UnknownValueError:
#    print("Google could not understand audio")
#except sr.RequestError as e:
#    print("Google error; {0}".format(e))
#    
    
    
    



































def wer(ref, hyp ,debug=False):
    
    
    DEL_PENALTY=1
    INS_PENALTY=1
    SUB_PENALTY=1
    
    r = ref.split()
    h = hyp.split()
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
 
    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
     
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL
         
    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS
     
    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1
                 
                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL
                 
    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns, 'Del':numDel}

