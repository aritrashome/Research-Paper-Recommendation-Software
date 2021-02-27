import csv
import re
import os.path
import glob
import sys
import pandas as pd
from ast import literal_eval as make_tuple
from operator import itemgetter

def findFile( InpFile ) :
    ALL = glob.glob("*.*")
    CSVfiles = glob.glob("*.csv")
    if InpFile not in ALL:
        raise SystemExit("\nError.File doesn't exist! \n")
    elif InpFile not in CSVfiles:
        raise SystemExit("\nError.I need a csv file! \n")    


if __name__ == "__main__":
    print("\n..............................Running rank_papers.py ....................................\n")
    fileName = "keywords2.csv"
    findFile(fileName)

    results = []
    with open(fileName) as File:
        reader = csv.reader(File)
        for row in reader:
            results.append(row)
        #print((type(results[0])))

    '''
    for row in results[:5]:
	    print row
    '''

    arfile = "Assoc_rules.csv"
    findFile(arfile)
    Rules = []
    with open(arfile) as File:
        reader = csv.reader(File)
        for row in reader:
            if(len(row)<3):
                continue
            Rules.append(row)
            #print(len(row))

    Rules_v2 = [ [ list(make_tuple(rule[0])) , list(make_tuple(rule[1])) , float(rule[2]) ]  for rule in Rules]

    match_rules= []
    count = 0
    for paper in results:
	    nums = []
	    for rule_index in range(0,len(Rules_v2)):
		    rule = Rules_v2[rule_index]
		    combined = rule[0]+rule[1]
		    #print rule[2]
		    if set(combined) <= set(paper):
		        #print (combined)
		        #print (paper)
		        nums.append(rule_index)
	    match_rules.append(nums)

    #print((len(results)))
    #print((len(match_rules)))

    result = []
    with open("keywords.csv") as keyfile:
        reader = csv.DictReader(keyfile)
        for row in reader:
            result.append(row)
        #print((type(result[0])))

    ranks =[]
    for paper,nums in zip(result,match_rules):
        mydict ={}
        mydict['No'] = paper.get('No',None)
        mydict['Source title'] = paper.get('Source title',None)
        mydict['Link'] = paper.get('Link',None)
        mydict['Rank'] =0 
        sum = 0
        for num in nums:
            sum = sum + Rules_v2[num][2]
        mydict['Confidence'] = sum
        #print((mydict['Confidence'] ))
        ranks.append(mydict)


    rankwise = sorted(ranks, key=itemgetter('Confidence'), reverse=True)
    rank=1
    for i in range(0,len(rankwise)):
        rankwise[i]['Rank'] = rank
        rank = rank + 1

    myF2 = open('paper_ranks.csv','w')
    with myF2:
        fieldNames = [ 'No','Source title' , 'Link' , 'Rank' , 'Confidence']
        writer = csv.DictWriter(myF2,fieldnames=fieldNames)
        writer.writeheader()
        for row in  rankwise:
            writer.writerow(row)      


    print('check paper_ranks.csv file for ranks')

			

