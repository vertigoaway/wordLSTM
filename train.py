import torch
from torch import nn
from torch.utils.data import DataLoader
import csv
import numpy.typing as npt
import numpy as np
from typing import Any
import trinketbox.ai.utils.tokenDataset as integerDataset
import trinketbox.ai.utils.NNLoops as loops
import wordLSTM
from griot import numpyWord as griotNPW
from griot import tool

BATCHSIZE = 32
epochs : int = 10
trainingDataPath : str = "discordData.csv"
lossFn = nn.CrossEntropyLoss(ignore_index=0)
learning_rate : float = 5e-4
l2Decay : float = 0




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


def loadTrainAndTestData(batch_size,vocab,dataReadout,trainingDataPath="data.csv",divisor=2,csvPosition=-3) -> tuple[DataLoader[Any], DataLoader[Any]]:

    ### Begin tokenizing data
    xraw = tool.flattenLines(vocab.tokenizeLines(dataReadout))
    x = np.array(xraw,dtype=np.int32)
    del xraw
    train_dataSet = integerDataset.lazyTextDataset(inSize=wordLSTM.inSize,outSize=wordLSTM.outSize,
                                    tokenizedData=x[0:len(x)//divisor],
                                    vocSize=len(vocab))
    test_dataSet = integerDataset.lazyTextDataset(inSize=wordLSTM.inSize,outSize=wordLSTM.outSize,
                                    tokenizedData=x[len(x)//divisor:],
                                    vocSize=len(vocab))
    train_dataloader = DataLoader(train_dataSet, batch_size=batch_size, 
                                shuffle=True,
                                num_workers=4,pin_memory=True,prefetch_factor=10)
    test_dataloader = DataLoader(test_dataSet, batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)
    return train_dataloader,test_dataloader

def main() -> None:

    dataReadout, rawData = pullWords(endMsgChar='[END]')

    vocab = initVocabulary(dataReadout,n=5000)


    train_dataloader, test_dataloader = loadTrainAndTestData(batch_size=BATCHSIZE,
                                                             vocab=vocab,
                                                             dataReadout=rawData)
    wordLSTM.vocab = vocab

    model = wordLSTM.create()
    try:
        model.loadWeights()
    except:
        pass
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=learning_rate,
                                 weight_decay=l2Decay)


    loopdeloop = loops.trainAndTest(train_dataloader,
                                        test_dataloader,
                                        model,
                                        lossFn,
                                        optimizer)
    epoch = 1
    while epoch<=10:
        print(f'epoch: {epoch}')
        try:
            loopdeloop.train_loop()
        except KeyboardInterrupt:
            model.saveWeights()
            return
        model.saveWeights()
        epoch+=1
    return

if __name__=='__main__':
    main()