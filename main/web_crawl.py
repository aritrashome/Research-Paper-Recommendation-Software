#URL PDF Parser
import urllib
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import BytesIO, StringIO


def pdf_from_url_to_txt(url):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr,  laparams=laparams)
    f = urllib.request.urlopen(url).read()
    fp = BytesIO(f)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    for page in PDFPage.get_pages(fp,
                                  pagenos,
                                  maxpages=maxpages,
                                  password=password,
                                  caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    str = retstr.getvalue()
    retstr.close()
    return str

import requests
from bs4 import BeautifulSoup
import re
import sys

query = str(sys.argv[1])
print(query)
#query = input("Enter query :")
urls = ["https://scholar.google.com/scholar?start=0&q="+str(query)+"&hl=en&as_sdt=0,5","https://scholar.google.com/scholar?start=10&q="+str(query)+"&hl=en&as_sdt=0,5","https://scholar.google.com/scholar?start=20&q="+str(query)+"&hl=en&as_sdt=0,5"]

data = []
for url in urls:
    try:
        page = requests.get(url)
    except: 
        continue
    soup = BeautifulSoup(page.content, 'html.parser')

    for item in soup.select('[data-did]'):
        paper = []  #1.Pdf link  2.title  3.Keywords(10) 4.body
        try:
            #print(str(item.select('div.gs_ggsd')[0].select('a')[0]['href']))  #Pdf link
            if(re.search('.pdf',item.select('div.gs_ggsd')[0].select('a')[0]['href'])):
                print(str(item.select('div.gs_ggsd')[0].select('a')[0]['href']))  #Pdf link
                paper.append(str(item.select('div.gs_ggsd')[0].select('a')[0]['href']))
            else: continue
        except:
            continue
        
        print(item.select('h3')[0].get_text()) #Title
        paper.append(item.select('h3')[0].get_text())
        try:
            text = pdf_from_url_to_txt(paper[0])
        except:
            continue

        # importing modules 
        from nltk.stem import PorterStemmer 
        from nltk.tokenize import word_tokenize 
        regex = re.compile('[^a-zA-Z.,0-9]')
        text = regex.sub(' ',text)
        #print(text)

        from rake_nltk import Rake
        from nltk.corpus import stopwords

        r = Rake(punctuations=".,"'',stopwords=stopwords.words('english'),max_length=1) # Uses stopwords for english from NLTK, and all puntuation characters.

        r.extract_keywords_from_text(text)

        tmp = r.get_ranked_phrases()[:5] # To get keyword phrases ranked highest to lowest.
        keyword = "; ".join(tmp)
        
        paper.append(keyword)
        try:
            paper.append(text[:1000])
        except:
            paper.append(text)
        data.append(paper)
        #print(item.select('.gs_rs')[0].get_text())   #Abstract
        #print(item.select('.gs_a')[0].get_text())   #Green text
        print('----------------------------')


import pandas as pd
df = pd.DataFrame(data,columns=['Link','Source title','Index Keywords','Abstract'])
df.to_csv('data.csv')

########################### KDM ##########################
import subprocess
import csv
subprocess.run('python kdm.py',shell=True)
fileName="paper_ranks.csv"

     
with open(fileName,encoding='latin-1') as File:
    reader = csv.DictReader(File)
    results = [ row for row in reader ]
#print(results[0],"\n",results[1])

df['KDM'] = 0
for row in results:
    idx = row.get('No',None)
    conf = row.get('Confidence',None)
    #print(idx,' ',conf,'\n')
    try:
        df['KDM'][int(idx)] = float(conf)
    except:
      continue

################################################ CAOT #################################
# coding: utf-8

# In[2]:

#All the header files required by the file
import numpy as np
import csv as csv
import pandas as pd
from scipy import stats as st
#import matplotlib.pyplot as plt
import math
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time
#from selenium.webdriver import browser
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from difflib import SequenceMatcher
import datetime
from selenium.webdriver.firefox.options import Options
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
#get_ipython().magic('matplotlib inline')


# In[3]:

#The name of the csv file to be opened
#readDataFileLoc = "scopus_2000_Citation.csv"


# In[4]:

#Reading the csv file and doing the required preprocessing
#It takes only the required columns - Authors, Title, Year, Cited by
#Removing the rows i.e. papers whose citations are 0
citationDataset = pd.read_csv("data.csv",)
#citationDataset=citationDataset1.iloc[:10,:]
citationDataset.head()
#newCitationDataset = citationDataset[['Authors', 'Title', 'Year', 'Cited by']]
#print(newCitationDataset.head())
#finalCitationDataset = newCitationDataset[pd.notnull(newCitationDataset['Cited by'])]
#listCitationData = finalCitationDataset.values
#print(len(listCitationData))
#print(finalCitationDataset)


# In[9]:

def getScholarSearchResult(browser, queryToSearch):
    """
    To type the query in the google scholar search bar and search it

    Args:
        browser: An open browser using selenium(preferably firefox)
        queryToSearch: A string which has to be searched in the google scholar

    Returns:
        browser(no need actually)
    """
    scholarSearchBar = browser.find_element_by_name("q")
    scholarSearchBar.send_keys(queryToSearch)
    scholarSearchBar.submit()
    return browser

def searchScholarResults(browser, queryToSearch):
    """
    To type the query in the google scholar search bar and search it

    Args:
        browser: An open browser using selenium(preferably firefox)
        queryToSearch: A string which has to be searched in the google scholar

    Returns:
        Nothing
    """
    scholarSearchBar = browser.find_element_by_name("q")
    scholarSearchBar.send_keys(queryToSearch)
    scholarSearchBar.submit()
    return

def customSimilarityChecker(listpapers, paperTitle):
    """
    A basic similarity checker which compares all the Names of paper displayed by the Scholar with the desired paper and clicks the link of the paper

    Args:
        listpapers: HTML listofpapers which contain the Title of the papers
        paperTitle: Title of the paper which we are searching

    Returns:
        Nothing
    """
    index = 0
    maxDiff = SequenceMatcher(None, listpapers[0].text, paperTitle).ratio()
    #print(maxDiff)
    for x in range(0, len(listpapers)):
        temp = SequenceMatcher(None, listpapers[x].text, paperTitle).ratio()
        if temp > maxDiff:
            index = x
            maxDiff = temp
    listpapers[index].click()
    return

def getYearCitationInformation(browser):
    """
    Calling this function will return an 1-d array containing pairs of year and citations in that year of the paper
    Takes the open page of the paper in Google Scholar as input

    Args:
        browser: An open browser using selenium(preferably firefox)

    Returns:
        A 1-d array containing pairs of year and citations in that year of the paper
    """
    answer = []
    try:
        idBars = browser.find_element_by_id('gsc_vcd_graph_bars')
    except Exception:
        return answer
    barsList = idBars.find_elements_by_tag_name('a')
    lengthBars = len(barsList)
    listOfBarsYear = idBars.find_elements_by_class_name('gsc_vcd_g_t')
    for i in range(0, lengthBars):
        answer.append([(barsList[lengthBars - i - 1].find_elements_by_tag_name('span'))[0].get_attribute('textContent'),
                       listOfBarsYear[lengthBars - i - 1].text])
    
    return answer

#This function will click the more button repeteadly to display all the research papers published by a person on the page
def correctClickMoreButton(browser):
    """

    Args:
        browser: 

    :return:
    """
    return

def searchChooseCitationInfo(browser, queryToSearch, paperTitle):
    """
    Calling this function will return an 1-d array containing pairs of year and citations in that year of the paper,
     taking only an open browser as input
    :param browser: An open browser using selenium(preferably firefox)
    :param queryToSearch: The text query which we are going to search in the google scholar
    :param paperTitle: Title of the paper which we are searching
    :return: A 1-d array containing pairs of year and citations in that year of the paper
    """
    browser.get("https://scholar.google.com/")
    scholarSearchBar = browser.find_element_by_name("q")
    scholarSearchBar.send_keys(queryToSearch)
    scholarSearchBar.submit()
    #tempList = browser.find_elements_by_class_name("gs_a")
    #tempList2 = tempList[0].find_elements_by_tag_name('a')
    #tempList2[0].click()
    time.sleep(5)
    ((browser.find_elements_by_class_name("gs_a"))[0].find_elements_by_tag_name('a'))[0].click()
    time.sleep(5)
    correctClickMoreButton(browser)
    listpapers = browser.find_elements_by_class_name('gsc_a_at')
    customSimilarityChecker(listpapers, paperTitle)
    time.sleep(5)
    return getYearCitationInformation(browser)




def writeListCitationToCsv(fileName, answer, listCitationData, score):
    """
    Writes the citation data to the csv file

    Args:
        fileName: name of the file to which data has to be written
        answer: citation data for each paper with respect to time
        listCitationData: consist of related data of each paper
        score: score for each paper

    Returns:
        Nothing
    """
    max = 0
    now = datetime.datetime.now()
    start = int(now.year)
    i = 0
    for i in range(0, len(answer)):
        if len(answer[i]) > 0:
            t = start - int(answer[i][0][1]) + len(answer[i])
        else:
            t = len(answer[i])
        if t > max:
            max = t
    f = open(fileName, 'w+')
    csvout = csv.writer(f)
    row = []
    row.extend(['Authors', 'Title', 'Year',
                'Cited by', 'Cited by updated', 'Last Citation Year', 'Score'])
    rowInitData = len(row)
    for i in range(0, max):
        row.extend([start - i])
    end = start - max + 1
    csvout.writerow(row)
    for i in range(0, len(answer)):
        row = []
        sum = 0
        for m in range(0, len(answer[i])):
            sum = sum + int(answer[i][m][0])
        if len(answer[i]) > 0:
            row.extend([listCitationData[i][0], listCitationData[i][1], listCitationData[i][2]
                           , listCitationData[i][3], sum, answer[i][len(answer[i]) - 1][1], score[i]])
        else:
            row.extend([listCitationData[i][0], listCitationData[i][1], listCitationData[i][2]
                           , listCitationData[i][3], 0, -1, score[i]])
        j = 0
        k = start
        while ((j < len(answer[i])) and (k >= end)):
            if (int(answer[i][j][1]) == k):
                row.extend([answer[i][j][0]])
                j = j + 1
            else:
                row.extend([0])
            k = k - 1
        incompIter = max - len(row) + rowInitData
        for l in range(0, incompIter):
            row.extend([0])
        csvout.writerow(row)
    return


# In[6]:

#Opening the browser
#self.driver = webdriver.Firefox()
#opts = webdriver.chrome.options.Options()
#opts.set_headless()
#assert opts.headless
browser = webdriver.Chrome("chromedriver.exe")
#browser = webdriver.Chrome(options=opts)
browser.get("https://scholar.google.com/")

# In[7]:

#Collect the data
answer = []
for i in range(0, len(df)):
    paperTitle = df["Source title"][i]
    
    #paperAuthor = listCitationData[i][0]
    try:
        answer.append(searchChooseCitationInfo(browser, paperTitle , paperTitle))
    except Exception:
        answer.append([])
    #print(i)
browser.quit()
#print(len(answer))


# In[30]:
score = []
def slopeMethodScoreRegression(answer, decay):
    """
    Returns a score telling how relevant the paper is from its citation data

    Args:
        answer: Citation data of all the papers
        decay: rate of decay of importance of citation data with respect to time

    Returns:
        score for every paper
    """
    now = datetime.datetime.now()
    currentYear = int(now.year)
    
    for i in range(0, len(answer)):
        sum = 0
        if len(answer[i]) > 0:
            sum = 0
            #sum = sum + int(answer[i][len(answer[i]) - 1][0])
            #sum = sum + int(answer[i][0][0]) * int(answer[i][0][0])
        else:
            score.extend([-1000])
            continue
        lenreq = len(answer[i])
        for j in range(0, lenreq - 1):
            diff = int(answer[i][lenreq - j - 2][0]) - int(answer[i][lenreq - j - 1][0])
            sum = diff * abs(diff) + sum * decay
        sum = sum + int(answer[i][len(answer[i]) - 1][0]) * int(answer[i][len(answer[i]) - 1][0])
        sum = sum + int(answer[i][0][0]) * int(answer[i][0][0])
        avgScore = sum / (currentYear - int(answer[i][len(answer[i]) - 1][1]) + 1)
        score.extend([avgScore])
    return score

score = slopeMethodScoreRegression(answer, 0.4)
df["CAOT"] = score
#citationDataset.to_csv("data.csv")
#print(score)
#score.append(score)
#x=listCitationData.assign(CA=pd.Series(score))
#print(x.head())
#x.to_csv('scopus_mod_new4.csv')
#writeListCitationToCsv('ansCit.csv', answer, listCitationData, score)
print("Done CAOT")

################### SQM ###########################
def SQM(answer):
    score = []
    for i in range(len(df)):
        sqm = 0
        if len(answer[i]):
            for j in range(len(answer[i])-1):
                slope = int(answer[i][j][0]) - int(answer[i][j+1][0])
                amp = abs(slope)
                sqm += slope*amp
            
        else:
            sqm = -1000
        score.append(sqm)
    return score

df['SQM'] = 0
df['SQM'] = SQM(answer)
print("Done SQM")

############################################### SCA ################################
import nltk
import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
from nltk.corpus import treebank
from nltk import FreqDist
from nltk.corpus import brown

nltk.download('brown')
nltk.download('cmudict')
nltk.download('averaged_perceptron_tagger')

#abstract="Among the many issues related to     data stream applications, those involved in predictive tasks such as classification and regression, play a significant role in Machine Learning (ML). The so-called ensemble-based approaches have characteristics that can be appealing to data stream applications, such as easy updating and high flexibility. In spite of that, some of the current approaches consider unsuitable ways of updating the ensemble along with the continuous stream processing, such as growing it indefinitely or deleting all its base learners when trying to overcome a concept drift. Such inadequate actions interfere with two inherent characteristics of data streams namely, its possible infinite length and its need for prompt responses. In this paper, a new ensemble-based algorithm, suitable for classification tasks, is proposed. It relies on applying boosting to new batches of data aiming at maintaining the ensemble by adding a certain number of base learners, which is established as a function of the current ensemble accuracy rate. The updating mechanism enhances the model flexibility, allowing the ensemble to gather knowledge fast to quickly overcome high error rates, due to concept drift, while maintaining satisfactory results by slowing down the updating rate in stable concepts. Results comparing the proposed ensemble-based algorithm against eight other ensembles found in the literature show that the proposed algorithm is very competitive when dealing with data stream classification. © 2018 Elsevier B.V."
frequency_list = FreqDist(i.lower() for i in brown.words())

def abstract_complexity(abstract):
    #abstract=abstract.decode('utf-8')

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

    try:
        AvgWordsperSentence=Nw/Ns
        AvgSyllablesperWord=Nsy/Nw
        GulpeaseIndex= 89- 10*(Nc/Nw)+ 300*(Ns/Nw)
        UnderstandabilityIndex=100*(Navg+0.75*Nsimple+0.5*Nhard)/Nw
    except:
        AvgWordsperSentence=0
        AvgSyllablesperWord=0
        GulpeaseIndex= 0
        UnderstandabilityIndex= 0

    return UnderstandabilityIndex
    '''
    print ("Number of Sentences - ",Ns)
    print ("Number of Words - ",Nw)
    print ("Number of characters - ",Nc)
    print ("Avergae Number of Words per Sentence - ",AvgWordsperSentence)
    print ("Average Number of Syllables per Word - ",AvgSyllablesperWord)
    print ("Gulpease Index - ",GulpeaseIndex)
    print ("Chunk Index - ",ChunkIndex)
    print ("UnderstandabilityIndex - ",UnderstandabilityIndex)
    '''
#with open("data.csv") as File:
#    reader = csv.DictReader(File)
#    results = [ row for row in reader ]
#df = pd.read_csv("data.csv")
df["SCA"] = 0
    
for i in range(len(df)):
    df["SCA"][i] = abstract_complexity(df["Abstract"][i])

print("Done SCA")

################################ Topics #######################################
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords 
import re
from nltk.stem import PorterStemmer
from collections import Counter
import re
import pickle

with open("Topic/test.txt", "rb") as fp:   # Unpickling
    w2vec_lda = pickle.load(fp)
from gensim.models import Word2Vec
model = Word2Vec.load("Topic/word2vec.model")

df['Topic'] = 0
n_topics = 5

def topic_assign(string):
    string = string.lower()
    string = re.sub(r'\W+', ' ', string)
    string = word_tokenize(string)
    ps = PorterStemmer()
    string = [ps.stem(w) for w in string]
    #string
    res = np.zeros(n_topics)
    from sklearn.metrics.pairwise import cosine_similarity
    for w in string:
        if w not in model.wv.vocab:
            continue
        sim = np.zeros(n_topics)
        for j in range(len(w2vec_lda)):
            for k in range(len(w2vec_lda[j])):
                sim[j] = sim[j] + cosine_similarity( model.wv[w2vec_lda[j][k][0]].reshape(1,50), model.wv[w].reshape(1,50) )[0][0]*w2vec_lda[j][k][1]
        res[sim.argmax()] = res[sim.argmax()] + 1
    if res[res.argmax()]==0:
        return None
    return res.argmax() + 1

for i in range(len(df)):
    df['Topic'][i] = topic_assign(df['Source title'][i])
print("Done Topic assign")

df.to_csv("data.csv")