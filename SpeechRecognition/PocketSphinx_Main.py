# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:08:02 2018

@author: Sunil Kantilal
"""

#from nltk import pos_tag
#from nltk import word_tokenize
#from nltk.corpus import stopwords

import PocketSphinx as sr
import base64
import json
import os
import re
# obtain path to "english.wav" in the same folder as this script


DISCOVERY_URL = ('https://{api}.googleapis.com/$discovery/rest?'
                 'version={apiVersion}')


def readdirectory(directory):
    directorywithslash=  re.sub(r'[//]*$','\',directory)
    files=os.listdir(directory)
    return [directorywithslash+x for x in files]


def readreftranscript(file):
     try:
         return open(file, 'r').read()
     except IOError:
          print("File not found or path incorrect")



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


def readaudiobyte(audiofile):
    try:
       with open(audiofile, 'rb') as speech:
        return base64.b64encode(speech.read())
    except IOError:
        print("File not found or path incorrect")
    finally:
        print("Skipping")


def readaudio(audiofile):   
         
       r=sr.Recognizer()
       objaudio= sr.AudioFile(audiofile)
       with objaudio as source:
           r.adjust_for_ambient_noise(objaudio, duration=1)
           audio=r.record(source)
           return audio

def gettranscript(recogaudio,apitype):
    r=sr.Recognizer()  
    if(apitype=='googlewebapi'):
        return r.recognize_google(recogaudio)
    elif(apitype=='bing'):
        return r.recognize_bing(recogaudio)
    elif(apitype=='googlecloud'):
        return r.recognize_bing(recogaudio)
    elif(apitype=='ibm'):
        return r.recognize_ibm(recogaudio)
    elif(apitype=='sphinx'):
        return r.recognize_sphinx(recogaudio,language='en-us')
    return

def validspeechfile(fileitem):
    """
    Filter invalid files from the directory before proceeding to process
    """
    if (re.search(r'(.[^.]+)$',fileitem).group(1) in ['.wav','.flac']) :
        print('Printing...',re.search(r'(.[^.]+)$',fileitem).group(1))
        return True
    else :
        print("Ignoring file since invalid:%s"%fileitem)
        return False

def getfileattribute(attrtype, fileitem):
    if(attrtype=='filename'):
        return re.search(r'\w+(?:\.\w+)*$',fileitem).group(0)
    else :
        return


SPEECH_INLOC='D:\\Training Files'
JSON_OUTLOC='D:\\audiojsonstore\\'
speechlist=[]

dictlist=readdirectory(SPEECH_INLOC)

for fileloc in filter(validspeechfile,dictlist):
     speechattr={}
     audioobj=readaudio(fileloc)
     speechattr['fileloc']=fileloc
     speechattr['filename']=getfileattribute('filename',fileloc)
     --speechattr['audiobase64']=readaudiobyte(fileloc).decode('utf8')
     speechattr['hypscript']=gettranscript(audioobj,'sphinx')
     speechattr['refscript']=readreftranscript(speechattr['fileloc']+'.txt')
     speechattr['WER']=wer(speechattr['hypscript'],speechattr['refscript'])
     speechlist.append(speechattr)



for audiojson in speechlist :
    j=json.dumps(audiojson, indent=4)
    print(JSON_OUTLOC+audiojson['filename']+'.json')
    f=open(JSON_OUTLOC+audiojson['filename']+'.json', 'w')
    print(j,file=f)
