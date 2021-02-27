import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
from nltk.corpus import treebank
from nltk import FreqDist
from nltk.corpus import brown

print "Give Abstract"
abstract=raw_input()
#abstract="Among the many issues related to     data stream applications, those involved in predictive tasks such as classification and regression, play a significant role in Machine Learning (ML). The so-called ensemble-based approaches have characteristics that can be appealing to data stream applications, such as easy updating and high flexibility. In spite of that, some of the current approaches consider unsuitable ways of updating the ensemble along with the continuous stream processing, such as growing it indefinitely or deleting all its base learners when trying to overcome a concept drift. Such inadequate actions interfere with two inherent characteristics of data streams namely, its possible infinite length and its need for prompt responses. In this paper, a new ensemble-based algorithm, suitable for classification tasks, is proposed. It relies on applying boosting to new batches of data aiming at maintaining the ensemble by adding a certain number of base learners, which is established as a function of the current ensemble accuracy rate. The updating mechanism enhances the model flexibility, allowing the ensemble to gather knowledge fast to quickly overcome high error rates, due to concept drift, while maintaining satisfactory results by slowing down the updating rate in stable concepts. Results comparing the proposed ensemble-based algorithm against eight other ensembles found in the literature show that the proposed algorithm is very competitive when dealing with data stream classification. © 2018 Elsevier B.V."
frequency_list = FreqDist(i.lower() for i in brown.words())

def abstract_complexity(abstract):
    abstract=abstract.decode('utf-8')

    sentences=sent_tokenize(abstract)
    Ns=len(sentences)

    d = cmudict.dict()
    punctions=[u'.',u',',u'?',u'!',u'(',u')',u'"',u';',u':',u'@',u'#',u'$',u'%',u'^',u'&',u'*',u'{',u'}',u'[',u']']
    Nw=0
    Nc=0
    Nsy=0
    Nhard=0
    Nsimple=0
    Navg=0

    for s in sentences:
        words=word_tokenize(s)
        for w in words:
            Nc=Nc+len(w)
            val=frequency_list[w]
            if w not in punctions:
                if val>40000:
                    Nsimple=Nsimple+1
                else:
                    if val<5000:
                        Nhard=Nhard+1
                    else:
                        Navg=Navg+1
            try:
                sy=[len(list(y for y in x if y[-1].isdigit())) for x in d[w.lower()]]
            except:
                sy=[0]
            Nsy=Nsy+sy[0]
        Nw=Nw+len(words)
        for p in punctions:
            Nw=Nw-words.count(p)

    sentences = nltk.sent_tokenize(abstract)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    Nch=0

    def chunk(sentence):
        chunkToExtract = """
        NP: {<NNP>*}
            {<DT>?<JJ>?<NNS>}
            {<NN><NN>}"""
        parser = nltk.RegexpParser(chunkToExtract)
        result = parser.parse(sentence)
        N=0
        for subtree in result.subtrees():
            N=N+1
        return N


    for sentence in sentences:
        Nch=Nch+chunk(sentence)

    Nw=float(Nw)
    Ns=float(Ns)
    Nc=float(Nc)
    Nsy=float(Nsy)
    Nch=float(Nch)
    Navg=float(Navg)
    Nhard=float(Nhard)
    Nsimple=float(Nsimple)

    AvgWordsperSentence=Nw/Ns
    AvgSyllablesperWord=Nsy/Nw

    GulpeaseIndex= 89- 10*(Nc/Nw)+ 300*(Ns/Nw)

    if Nch==1:
        ChunkIndex=100
    else:
        ChunkIndex=100/((Nch/Ns)-1)

    UnderstandabilityIndex=100*(Navg+0.75*Nsimple+0.5*Nhard)/Nw

    print "Number of Sentences - ",Ns
    print "Number of Words - ",Nw
    print "Number of characters - ",Nc
    print "Avergae Number of Words per Sentence - ",AvgWordsperSentence
    print "Average Number of Syllables per Word - ",AvgSyllablesperWord
    print "Gulpease Index - ",GulpeaseIndex
    print "Chunk Index - ",ChunkIndex
    print "UnderstandabilityIndex - ",UnderstandabilityIndex

abstract_complexity(abstract)