from difflib import SequenceMatcher
import csv
import re
import os.path
import glob
import sys
from nltk.corpus import stopwords
from operator import itemgetter 

import subprocess


def findFile( InpFile ) :
    ALL = glob.glob("*.*")
    CSVfiles = glob.glob("*.csv")
    if InpFile not in ALL:
        raise SystemExit("\nError.File doesn't exist! \n")
    elif InpFile not in CSVfiles:
        raise SystemExit("\nError.I need a csv file! \n")         

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio() 

if __name__ == "__main__":

    print("\n..............................Running apriori_preprocess.py ....................................\n")

    stop_words = set(stopwords.words('english'))

    
    fileName = input("Give a  corpus file (.csv file required) : ")
    findFile(fileName)
     
    # read the contents of the corpus file. 
    with open(fileName) as File:
        reader = csv.DictReader(File)
        results = [ row for row in reader ]


    dict_keywords = {}  # paperNum is the key
    paperNum = 0
    keydict=[]    
    keywordslist = []
    allwords =[]
    for row in results:
        title = row.get('Source title',None)
        link = row.get('Link',None)
        i_keystring = row.get('Index Keywords',None)
        indexkeylist = [ string.strip().lower() for string in re.split(';',i_keystring) ]
        trimmed2 = [ string for string in indexkeylist if string and not string in stop_words ]
        if len(trimmed2):
            mydict = {}
            mydict['No'] = paperNum
            mydict['Source title'] = title
            mydict['Link'] = link
            mydict['Keywords'] = trimmed2
            dict_keywords[paperNum] = mydict
            keydict.append(mydict)
            keywordslist.append(trimmed2)
            for q in range(0,len(trimmed2)):
                word = [len(keywordslist)-1,q,trimmed2[q]]
                allwords.append(word) 
        paperNum = paperNum + 1        

    #print((len(keywordslist)))
    #print((len(keywordslist[0])))
    #process all words
    #print((len(allwords)))
    allwords = sorted(allwords, key=itemgetter(2))
    classes = []
    curr_c_ind = -1
    currentclass = []
    for k in range(0,len(allwords)):
        word = allwords[k]
        #print len(word)
        if k!=0 :
            prev = allwords[k-1]
            sim = similar(prev[2],word[2])
            if sim<0.5:
                currentclass = []
                currentclass.append(word)
                classes.append(currentclass)
                curr_c_ind = curr_c_ind + 1
            else:
                classes[curr_c_ind].append(word) 
            #print word          
        elif k==0:
            currentclass = []
            currentclass.append(word)
            classes.append(currentclass)
            curr_c_ind = 0

    for clas in classes:
        # the first word of the class
        comword = clas[0][2]
        for word in clas:
            papnum = word[0]
            ind = word[1]
            #print papnum,ind
            keywordslist[papnum][ind] = comword   
    
    myF2 = open('keywords_for_viewing.csv','w')
    with myF2:
        fieldNames = [ 'No','Source title' , 'Link' , 'Keywords']
        writer = csv.DictWriter(myF2,fieldnames=fieldNames)
        writer.writeheader()
        for row in  keydict:
            writer.writerow(row)  

    myFile = open('keywords2.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(keywordslist)
         
    print("Writing complete")
    print ("The keywords of each paper are present in 'keywords2.csv'")
    print ("Give this file to apriori.py for finding the association rules"   )
    
    print("\nDo you want to give custom support and confidence thresholds for Association rules? (y/n) ")
    print("NOTE : No takes default values : Support - 0.15 , Confidence - 0.6")
    
    option = input()
    if option=="y":
        s = str( input("Support value : ") )
        c = str( input("Confidence value : ") )
    else:
        s = ""    
        c = ""
        
    makestring = "python apriori.py -f keywords2.csv" + " " + s + " " + c
    
    subprocess.run(makestring,shell=True)
    subprocess.run('python rank_papers.py',shell=True)
