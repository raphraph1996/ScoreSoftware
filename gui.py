from tkinter import *
from tkinter.filedialog import *
from tkinter import messagebox
import json
import re
from string import digits
from gensim.models import Doc2Vec
from nltk.stem.snowball import FrenchStemmer
import nltk
import numpy as np
import unidecode
from scipy import spatial
import math
import operator
hiddenimports = ["nltk.chunk.named_entity"]
regexURL = r"""(?i)((?:(?:(https?|s?ftp)):\/\/)?(?:www\.)?(?:(?:(?:[A-Z0-9][A-Z0-9-]{0,61}[A-Z0-9]\.)+)(?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)|(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))(?::(\d{1,5}))?(?:(?:\/\S+)*))"""
regexMail = r"""(?i)[\w\.-]+@[\w\.-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)"""


def computetitleScore(title):
    titleS = nltk.word_tokenize(re.sub('\W+',' ',title))
    titleS = [''.join([i for i in word if not i.isdigit()]) for word in titleS]
    titleS = [word for word in titleS if word.lower() not in list(line.strip() for line in open('french')) and len(word)>1]
    with open('titleVar.json','r') as input:
        titlesVar = json.load(input)
    input.close()
    if len(titleS) == 0:
        score = 0
    else:
        if title not in titlesVar.keys():
            if len(titleS) == 1:
                score = 0.5
            elif len(titleS) == 2:
                score =1
            else:
                score=2
        elif titlesVar[title] < 51:
            if len(titleS) == 1:
                score = 0.5
            elif len(titleS) == 2:
                score =1
            else:
                score=2
        elif titlesVar[title] < 311:
            if len(titleS) == 1:
                score = 0.25
            elif len(titleS) == 2:
                score = 0.75
            else:
                score= 1.5
        else:
            score=2
    return score

def computedescScore(desc):
    model = Doc2Vec.load('doc2vec')
    vec_infer = model.infer_vector(desc)
    tot = 0
    for i in range(len(model.docvecs)):
        tot += spatial.distance.cosine(vec_infer,model.docvecs[i])
    return ((tot/len(model.docvecs)) - 0.38)/(1.23-0.38)


def computecatScore(desc,cat):
    with open('cat','r') as input:
        listCat = json.load(input)
    input.close()
    with open('probaBayes.json','r') as input:
        proba=json.load(input)
    input.close()
    probs = []
    for catP in listCat:
        if catP not in proba['priors']:
            continue
        prob = proba['priors'][catP]
        for word in desc:
            if word not in proba['likehoods'][catP]:
                prob *= proba['likehoods0'][catP]
            else:
                prob *= proba['likehoods'][catP][word]
        probs.append((catP,prob))
    probs.sort(key = operator.itemgetter(1), reverse = True)
    for i in range(11):
        if i > 9:
            return 0.1
        else :
            if probs[i][0] == cat:
                if i >4:
                    return 0.2+0.1*(9-i)
                elif i ==4:
                    return 0.75
                elif i ==3:
                    return 0.9
                else:
                    return 1



def computeScores(title,desc,cat):
    if title == "":
        messagebox.showwarning("Empty attribute","Empty title. Title Score automatically set to 0.")
        titlescore = 0
    else:
        titlescore = computetitleScore(title)
        print(titlescore)

    if desc == "":
        messagebox.showwarning("Empty attribute","Empty description. Description and Category Score automatically set to 0.")
        descscore = 0
        links = 0
        categoryScore = 0
    else:
        urls = re.findall(regexURL,desc)
        mails = re.findall(regexMail,desc)
        words = nltk.word_tokenize(re.sub('\W+',' ',re.sub(regexMail,'',re.sub(regexURL,'', desc))))
        words = [''.join([i for i in word if not i.isdigit()]) for word in words]
        stemmer = FrenchStemmer()
        newwords = []
        for word in words:
            if word.lower() not in list(line.strip() for line in open('french')) and len(word)>1:
                if word.lower() in list(line.strip() for line in open('words')):
                    newwords.append(unidecode.unidecode(stemmer.stem(word.lower())))
                newwords.append(unidecode.unidecode(stemmer.stem(word.lower())))
        links = len(urls)+len(mails)
        descscore = computedescScore(newwords)
        if cat == None:
            categoryScore = 0
        else:
            categoryScore = computecatScore(newwords,cat)
    totScore = 7*descscore+categoryScore+titlescore
    if links > 0:
        totScore+=1
    if totScore > 10:
        totScore=10

    message = "The total score is "+str(round(totScore,2))+"\nThe description score is "+str(round(descscore,2))+"\nThe title score is "+str(titlescore)+"\nThe classification score is "+str(categoryScore)+"\n"
    if descscore < 0.4:
        message+="You should improve the description\n"
    if titlescore < 1:
        message+="You should improve the title\n"
    if categoryScore < 0.5:
        message+="The description should be more explcit on the category"
    messagebox.showinfo("Results",message)

def proces():
    title = Entry.get(E1)
    desc = Entry.get(E2)
    program_category = Entry.get(E3)
    program_contenttypestat = Entry.get(E4)

    with open('cat','r') as input:
        listCat = json.load(input)
    input.close()
    if program_category == "" and program_contenttypestat == "":
        messagebox.showwarning("Empty attribute","Empty program_category and program_contenttypestat. Classification Score automatically set to 0.")
        cat = None
    elif program_category == "" and program_contenttypestat != "":
        if program_contenttypestat in listCat:
            cat = program_contenttypestat
        else:
            messagebox.showwarning("Unknown Category","Unknown program_category and/or program_contenttypestat. Classification Score automatically set to 0.")
            cat = None
    elif program_category != "" and program_contenttypestat == "":
        if program_category in listCat:
            cat = program_category
        else:
            messagebox.showwarning("Unknown Category","Unknown program_category and/or program_contenttypestat. Classification Score automatically set to 0.")
            cat = None
    else:
        if program_category + "_" + program_contenttypestat in listCat:
            cat = program_category + "_" + program_contenttypestat
        elif program_contenttypestat + "_" + program_category in listCat:
            cat = program_contenttypestat + "_" + program_category
        else:
            if program_category in listCat:
                messagebox.showwarning("Unknown Category","Unknown program_contenttypestat. Only program_category will be used.")
                cat = program_category
            elif program_contenttypestat in listCat:
                messagebox.showwarning("Unknown Category","Unknown program_category. Only program_contenttypestat will be used")
                cat = program_contenttypestat
            else:
                messagebox.showwarning("Unknown Category","Unknown program_category and program_contenttypestat. Classification score automatically set to 0.")
                cat = None
    computeScores(title,desc,cat)


top = Tk()
L1 = Label(top, text="Attributes",).grid(row=0,column=1)
L2 = Label(top, text="object_title",).grid(row=1,column=0)
L3 = Label(top, text="object_desc",).grid(row=2,column=0)
L4 = Label(top, text="program_category",).grid(row=3,column=0)
L5 = Label(top, text="program_contenttypestat",).grid(row=4,column=0)
E1 = Entry(top, bd =5)
E1.grid(row=1,column=1)
E2 = Entry(top, bd =5)
E2.grid(row=2,column=1)
E3 = Entry(top, bd =5)
E3.grid(row=3,column=1)
E4 = Entry(top, bd =5)
E4.grid(row=4,column=1)
B=Button(top, text ="Submit",command = proces).grid(row=5,column=1,)


top.mainloop()
