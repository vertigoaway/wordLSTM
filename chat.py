import wordLSTM as wlstm
import trinketbox.ai.utils.outProcessing as post
from typing import Any
from griot import tool
from griot import numpyWord as griotNPW
import csv

def pullWords(trainingDataPath="discordData.csv",csvPosition=-3,endMsgChar:str="[END]") -> list[Any]:
    uncutReadout = []
    with open(trainingDataPath, "r") as csvfile:
        readout = list(csv.reader(csvfile))[1:]
        out : list[Any]= []
        for r in readout:
            if len(r[csvPosition]) > 3:
                out.append(r[csvPosition].strip().lower())
                uncutReadout.append(out[-1])
                out[-1] = out[-1].split(' ')# + [endMsgChar]
    readout = out
    return [readout,uncutReadout]

def initVocabulary(dataReadout, n=10000) -> griotNPW.StrictVocab:
    voc = griotNPW.StrictVocab()
    o = tool.flattenLines(dataReadout)
    ctr = tool.getWordCtr(o)

    commonWordsSet = ctr.most_common(n=n)
    commonWords = []
    for w in commonWordsSet:
        commonWords.append(w[0])
    try:
        commonWords.remove("[END]")
    except:
        pass
    voc.addWords(commonWords)
    return voc

#TODO: use command line args for choosing what weights to load


if __name__=='__main__':
    vocab = initVocabulary(pullWords()[0],n=5000)
    wlstm.vocab = vocab

    model = wlstm.create()
    model.loadWeights()
    print('starting terminal interface')
    post.basicInterface(model,vocab,timeSteps=wlstm.inSize)