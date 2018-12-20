#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:50:26 2018

@author: srinivas
"""

#All the required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import enchant
d = enchant.Dict("en_US")
from geniatagger import GeniaTagger
tagger = GeniaTagger('./geniatagger/geniatagger')
from sklearn.svm import SVC
from nltk import Tree


#Training 

# Data Extraction

with open('train.txt') as f:
    lines = f.readlines()
    
list_words=[]
for i in range(0, len(lines)):
    list_words.append(lines[i].split())

    
lists=pd.DataFrame(list_words)

sentence=""
main_word=[]
affix=[]
negation_words=[]
attached_text=[]


for i in range(0, len(lists)):
    if(str(lists[7][i])!='***'):
        if(str(lists[7][i])!='_'):
            if(str(lists[7][i])!='None'):
                sentence=sentence+str(lists[4][i]).lower()+" "
                if(str(lists[8][i]).isalpha()):
                    attached_text.append(str(lists[8][i]).lower())
                    affix.append(str(lists[7][i]).lower())
            else:
                if(sentence!=""):
                    sentence=sentence[:-1]
                    negation_words.append(sentence)
                    sentence=""    
                    
cleaned_affix=[]
cleaned_attached=[]
cleaned_negation=[]


[cleaned_affix.append(x) for x in affix if x not in cleaned_affix]                
                    
[cleaned_negation.append(x) for x in negation_words if x not in cleaned_negation]

[cleaned_attached.append(x) for x in attached_text if  x not in cleaned_attached]


words=[]
for i in range(0,len(lists)):
    if(str(lists[4][i])!='None'):
       if(str(lists[4][i]).isalpha()):
           if(str(lists[4][i]).lower() not in words):
               words.append(str(lists[4][i]).lower())
               
               
final_li=[]
for i in words:
    if i not in final_li:
            for j in cleaned_affix:
                if(i.find(j)!=-1):
                    final_li.append(i)
                    


#Cue Training
                    
final_list=[]
[final_list.append(x) for x in final_li if x not in final_list]
X=[]
inputs={}
for i in final_list:
    ''' this is feature 2'''
    flag1=0
    for j in cleaned_affix:
        if (i.find(j)!=-1):
            flag1=1
            a=i.split(j)
            word_tag=tagger.parse(i)[0][2]
            if(len(a[0])>1 or len(a[1])>1):
             if(len(a[0])>=len(a[1])):
                if wordnet.synsets(a[0]):
                    q=1
                else:
                    q=0
                ''' this is feature 3'''
                m=tagger.parse(a[0])[0][2]
                if m==word_tag:
                    r=1
                else:
                    r=0
                ''' this is  feature 4'''
                antonyms = [] 
                for syn in wordnet.synsets(i):
                  for l in syn.lemmas():
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name())    
                s=0
                for k in antonyms:
                    if(a[0].find(k)!=-1 or k.find(a[0])!=-1):
                        s=1
             else:
                 if wordnet.synsets(a[1]):
                    q=1
                 else:
                    q=0
                 if tagger.parse(a[1])[0][2]==word_tag:
                    r=1
                 else:
                    r=0
                 antonyms = [] 
                 for syn in wordnet.synsets(i):
                  for l in syn.lemmas():
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name())    
                 s=0
                 for k in antonyms:
                    if(a[1].find(k)!=-1 or k.find(a[1])!=-1):
                        s=1
             u=len(j)
             if(len(a[0])==0 and len(a[1])>0):
                t=1
             elif(len(a[0])>0 and len(a[1])>0):
                t=2
             elif(len(a[0])>0 and len(a[1])==0):
                t=3
             else:
                t=0
            else:
                q=0
                r=0
                s=0
                u=0    
                t=0
    if(flag1==0):
        q=0
        r=0
        s=0
        u=0
        t=0
    if(t==0):
      inputs[i]=[q,r,s,u,0,0,0] 
    elif(t==1):
      inputs[i]=[q,r,s,u,0,0,1] 
    elif(t==2):
      inputs[i]=[q,r,s,u,0,1,0] 
    else:
      inputs[i]=[q,r,s,u,1,0,0]
      
output={}                    
for i in final_list:
  if i in inputs.keys():
    if i in cleaned_negation:
        output[i]=1
    else:
        output[i]=0
    
y=[]  
X=[]             
for i in final_list:
    X.append(inputs[i])
    y.append(output[i])
            

classifier_cue = SVC(kernel = 'rbf', random_state = 0)
classifier_cue.fit(X,y)

#Scope Training
sentences=[]
sentence=""
phrases=[]
phrase=""
pos=[]
tag=""
for i in range(0, len(lists)):
  if(str(lists[7][i])!='***'):
    if(str(lists[0][i])!='None'):
        sentence+=str(lists[4][i]).lower()+" "
        phrase+=lists[6][i]+" "
        tag+=lists[5][i]+" "
    else:
        if(sentence!=""):
            sentences.append(sentence)
        sentence=""
        phrases.append(phrase)
        phrase=""
        pos.append(tag)
        tag=""
        
word=""        
word_phrase=[]
flag=0
#s3="^"
for i in range(0, len(lists)):
  if(str(lists[7][i])!='***'):
    if(str(lists[0][i])!='None'):
        word+=lists[6][i].replace("*"," "+str(lists[4][i]).lower())
    else:
        if(word!=''):
            word_phrase.append(word)
        word=""

scope=[]   
sc=[]  
sentence=""
count=0
for i in range(0, len(lists)):
  if(str(lists[0][i])!='None'):  
    if(str(lists[7][i])!='***'):
        if(str(lists[8][i]).isalpha()):
                scope.append(str(lists[8][i]).lower())
        if(str(lists[7][i])!='_'):
                sentence=sentence+str(lists[4][i]).lower()+" "
  else:
                if(sentence!=''):
                    sentence=sentence[:-1]
                    sc.append([sentence,scope])
                scope=[]
                sentence="" 
     
''' 
 left distance
 right ditance
 tree sidtance
 comma in path'''
inputs=[]   
remove=[',','" "','""','.','``','` `']      
for i in range(0,len(word_phrase)):
   tree=Tree.fromstring(word_phrase[i])
   traversal=tree.treepositions('leaves')
#   print(i)
   negative=sc[i][0]
   if(str(negative).find(" ")==-1):
    word=tree.leaves()
    negative_position=word.index(negative)
    for j in range(0,len(word)):
       if(j<negative_position):
           r_distance=0
           l_distance=negative_position-j
       elif(j>negative_position):
            l_distance=0
            r_distance=j-negative_position
       else:
            l_distance=0
            r_distance=0
       neg_tree_length=len(traversal[negative_position])
       neg_tree=traversal[negative_position]
       word_length=len(traversal[j])
       word_tree=traversal[j]
       if(neg_tree_length<=word_length):
            position=neg_tree_length
            for k in range(0,neg_tree_length):
                if(neg_tree[k]!=word_tree[k]):
                    position=k
                    break;
            distance=neg_tree_length-position
            distance+=word_length-position
       else:
            position=word_length
            for k in range(0,word_length):
                if(neg_tree[k]!=word_tree[k]):
                    position=k
                    break;
            distance=word_length-position
            distance+=neg_tree_length-position
       neg_pos_tag=tagger.parse(negative)[0][2]
       word_pos_tag=tagger.parse(word[j])[0][2]
       comma_count=0
       if j<=negative_position: 
         for k in range(j,negative_position):
           if(str(word[k])==','):
               comma_count+=1
       else:
          m=negative_position
          for k in range(m,j):
              if(str(word[k])==','):
                comma_count+=1
       if(word[j] in sc[i][1]):
          y=1
       else:
           y=0
       inputs.append([l_distance,r_distance,distance,comma_count,y])


inp=np.array(inputs)

X=inp[:,0:-1]
Y=inp[:,-1]
y=np.reshape(Y,(-1,1))

classifier_scope = SVC(kernel = 'rbf', random_state = 0)
classifier_scope.fit(X,y)


#Evaluation on Dev data

with open('dev.txt') as f:
    lines = f.readlines()
    
list_words=[]
for i in range(0, len(lines)):
    list_words.append(lines[i].split())


    
lists=pd.DataFrame(list_words)

cue_answer=[]
for w in range(len(lists)):
    if(str(lists[0][w])!='None'):
        i=str(lists[4][w])
        if i.lower() in cleaned_negation:
            cue_answer.append(i)
        else:
            flag1=0
            for j in cleaned_affix:
                if (i.find(j)!=-1):
                    flag1=1
                    a=i.split(j)
                    word_tag=tagger.parse(i)[0][2]
                    if(len(a[0])>1 or len(a[1])>1):
                        if(len(a[0])>=len(a[1])):
                            if wordnet.synsets(a[0]):
                                q=1
                            else:
                                q=0
                            ''' this is feature 3'''
                            m=tagger.parse(a[0])[0][2]
                            if m==word_tag:
                                r=1
                            else:
                                r=0
                            ''' this is  feature 4'''
                            antonyms = [] 
                            for syn in wordnet.synsets(i):
                                for l in syn.lemmas():
                                    if l.antonyms(): 
                                        antonyms.append(l.antonyms()[0].name())    
                            s=0
                            for k in antonyms:
                                if(a[0].find(k)!=-1 or k.find(a[0])!=-1):
                                    s=1
                        else:
                            if wordnet.synsets(a[1]):
                                q=1
                            else:
                                q=0
                            if tagger.parse(a[1])[0][2]==word_tag:
                                r=1
                            else:
                                r=0
                            antonyms = [] 
                            for syn in wordnet.synsets(i):
                                for l in syn.lemmas():
                                    if l.antonyms(): 
                                        antonyms.append(l.antonyms()[0].name())    
                            s=0
                            for k in antonyms:
                                if(a[1].find(k)!=-1 or k.find(a[1])!=-1):
                                    s=1
                        u=len(j)
                        if(len(a[0])==0 and len(a[1])>0):
                            t=1
                        elif(len(a[0])>0 and len(a[1])>0):
                            t=2
                        elif(len(a[0])>0 and len(a[1])==0):
                            t=3
                        else:
                            t=0
                    else:
                        q=0
                        r=0
                        s=0
                        u=0    
                        t=0
            if(flag1==0):
                q=0
                r=0
                s=0
                u=0
                t=0
            if(t==0):
                x=[q,r,s,u,0,0,0] 
            elif(t==1):
                x=[q,r,s,u,0,0,1] 
            elif(t==2):
                x=[q,r,s,u,0,1,0] 
            else:
                x=[q,r,s,u,1,0,0]
            x=np.array(x)
            x=x.reshape(1,-1)
            out=classifier_cue.predict(x)
            if out==1:
                cue_answer.append(i)
            else:
                cue_answer.append('_')
    else:
        cue_answer.append('@#$')
                          
i=0
while(i<len(lists)):
    count=0
    j=i
    while(j<len(lists)):
        if str(cue_answer[j]).isalpha():
            count=1
        if str(cue_answer[j])=='@#$':
               j=j+1
               break
        j=j+1
    if count==0:
        for k in range(i,j-1):
                        cue_answer[k] ='***'
    i=j
 
cue_answer[len(lists)-1]='***'            

i=0
scope_answer=[]
safty=scope_answer
while(i<len(lists)):
   if(cue_answer[i]=='***'):
           scope_answer.append('@')
           i=i+1
   elif(cue_answer[i]=='@#$'):
         scope_answer.append('@#$')
         i=i+1
   else:
        scope_answer.append(str(lists[4][i]))    
        i=i+1

i=0
while(i<len(lists)):
    if str(cue_answer[i])!='***' and str(cue_answer[i])!='@#$':
        sentence=""
        parser=""
        m=i
        scope=[]
        index=[]
        while(cue_answer[m]!='@#$'):
            if(str(cue_answer[m]).isalpha()):
                negative=cue_answer[m]
            sentence=sentence+scope_answer[m]+" "
            parser+=str(lists[6][m]).replace("*"," "+scope_answer[m])
            m=m+1
        tree=Tree.fromstring(parser)    
        traversal=tree.treepositions('leaves')
        word=tree.leaves()
        negative_position=word.index(negative)
        for j in range(0,len(word)):
         if(str(word[j]).isalpha()):
            if(j<negative_position):
                r_distance=0
                l_distance=negative_position-j
            elif(j>negative_position):
                l_distance=0
                r_distance=j-negative_position
            else:
                l_distance=0
                r_distance=0
            neg_tree_length=len(traversal[negative_position])
            neg_tree=traversal[negative_position]
            word_length=len(traversal[j])
            word_tree=traversal[j]
            if(neg_tree_length<=word_length):
                position=neg_tree_length
                for k in range(0,neg_tree_length):
                    if(neg_tree[k]!=word_tree[k]):
                        position=k
                        break
                distance=neg_tree_length-position
                distance+=word_length-position
            else:
                position=word_length
                for k in range(0,word_length):
                    if(neg_tree[k]!=word_tree[k]):
                        position=k
                        break
                distance=word_length-position
                distance+=neg_tree_length-position
            comma_count=0
            l=j
            if l<=negative_position:
                for p in range(l,negative_position):
                        if(str(word[p])==','):
                               comma_count+=1
            else:
                for p in range(negative_position,l):
                        if(str(word[p])==','):
                            comma_count+=1
            x=[l_distance,r_distance,distance,comma_count]
            x=np.array(x)
            x=x.reshape(1,-1)
            y=classifier_scope.predict(x)
            if(y==1):
                scope.append(word[j])
                flag3=0
        for j in range(i,m):
            if scope_answer[j] not in scope:
                    scope_answer[j]='_'
                       
        i=m
    else:
        i=i+1

for i in range(0,len(lists)):
    if(cue_answer[i].isalpha()):
        for j in cleaned_affix:
            flag1=0
            if(str(cue_answer[i]).find(j)!=-1):
                    flag1=1
                    a=cue_answer[i].split(j)
                    word_tag=tagger.parse(cue_answer[i])[0][2]
                    if(len(a[0])>1 or len(a[1])>1):
                        b=0
                        if(len(a[0])>=len(a[1])):
                            b=1
                            if wordnet.synsets(a[0]):
                                q=1
                            else:
                                q=0
                            ''' this is feature 3'''
                            m=tagger.parse(a[0])[0][2]
                            if m==word_tag:
                                r=1
                            else:
                                r=0
                            ''' this is  feature 4'''
                            antonyms = [] 
                            for syn in wordnet.synsets(cue_answer[i]):
                                for l in syn.lemmas():
                                    if l.antonyms(): 
                                        antonyms.append(l.antonyms()[0].name())    
                            s=0
                            for k in antonyms:
                                if(a[0].find(k)!=-1 or k.find(a[0])!=-1):
                                    s=1
                        else:
                            if wordnet.synsets(a[1]):
                                q=1
                            else:
                                q=0
                            if tagger.parse(a[1])[0][2]==word_tag:
                                r=1
                            else:
                                r=0
                            antonyms = [] 
                            for syn in wordnet.synsets(cue_answer[i]):
                                for l in syn.lemmas():
                                    if l.antonyms(): 
                                        antonyms.append(l.antonyms()[0].name())    
                            s=0
                            for k in antonyms:
                                if(a[1].find(k)!=-1 or k.find(a[1])!=-1):
                                    s=1
                        u=len(j)
                        if(len(a[0])==0 and len(a[1])>0):
                            t=1
                        elif(len(a[0])>0 and len(a[1])>0):
                            t=2
                        elif(len(a[0])>0 and len(a[1])==0):
                            t=3
                        else:
                            t=0
                        break
                    else:
                        q=0
                        r=0
                        s=0
                        u=0    
                        t=0
        if(flag1==0):
                q=0
                r=0
                s=0
                u=0
                t=0
        if(t==0):
                x=[q,r,s,u,0,0,0] 
        elif(t==1):
                x=[q,r,s,u,0,0,1] 
        elif(t==2):
                x=[q,r,s,u,0,1,0] 
        else:
                x=[q,r,s,u,1,0,0]
        x=np.array(x)
        x=x.reshape(1,-1)
        out=classifier_cue.predict(x)
        if out==1:
            if b==1:
                    scope_answer[i]=a[0]
                    cue_answer[i]=j
            else:
                    scope_answer[i]=a[1]
                    cue_answer[i]=j
                    
n=None                    
#Saving the result
final_sentences=[]                    
for i in range(len(lists)):
    sentence=""
    if( str(cue_answer[i])!='@#$'):
     if (str(cue_answer[i])!='***'):
        for j in range(0,7):
            sentence+=str(lists[j][i])+"	"
        sentence+=str(cue_answer[i])+"	"
        sentence+=str(scope_answer[i])+"	"
        sentence+=str('_')
     else:
        for j in range(0,7):
            sentence+=str(lists[j][i])+"	"
        sentence+=str(cue_answer[i])+"	"
    else:
        sentence+=""	
    final_sentences.append(sentence)

for i in final_sentences:
    text_file = open("dev_output.txt", "a")
    text_file.write("%s\n" % i)
    
    
    
    
# Test file 1
with open('test1.txt') as f:
    lines = f.readlines()
    
list_words=[]
for i in range(0, len(lines)):
    list_words.append(lines[i].split())


    
lists=pd.DataFrame(list_words)

cue_answer=[]
for w in range(len(lists)):
    if(str(lists[0][w])!='None'):
        i=str(lists[4][w])
        if i.lower() in cleaned_negation:
            cue_answer.append(i)
        else:
            flag1=0
            for j in cleaned_affix:
                if (i.find(j)!=-1):
                    flag1=1
                    a=i.split(j)
                    word_tag=tagger.parse(i)[0][2]
                    if(len(a[0])>1 or len(a[1])>1):
                        if(len(a[0])>=len(a[1])):
                            if wordnet.synsets(a[0]):
                                q=1
                            else:
                                q=0
                            ''' this is feature 3'''
                            m=tagger.parse(a[0])[0][2]
                            if m==word_tag:
                                r=1
                            else:
                                r=0
                            ''' this is  feature 4'''
                            antonyms = [] 
                            for syn in wordnet.synsets(i):
                                for l in syn.lemmas():
                                    if l.antonyms(): 
                                        antonyms.append(l.antonyms()[0].name())    
                            s=0
                            for k in antonyms:
                                if(a[0].find(k)!=-1 or k.find(a[0])!=-1):
                                    s=1
                        else:
                            if wordnet.synsets(a[1]):
                                q=1
                            else:
                                q=0
                            if tagger.parse(a[1])[0][2]==word_tag:
                                r=1
                            else:
                                r=0
                            antonyms = [] 
                            for syn in wordnet.synsets(i):
                                for l in syn.lemmas():
                                    if l.antonyms(): 
                                        antonyms.append(l.antonyms()[0].name())    
                            s=0
                            for k in antonyms:
                                if(a[1].find(k)!=-1 or k.find(a[1])!=-1):
                                    s=1
                        u=len(j)
                        if(len(a[0])==0 and len(a[1])>0):
                            t=1
                        elif(len(a[0])>0 and len(a[1])>0):
                            t=2
                        elif(len(a[0])>0 and len(a[1])==0):
                            t=3
                        else:
                            t=0
                    else:
                        q=0
                        r=0
                        s=0
                        u=0    
                        t=0
            if(flag1==0):
                q=0
                r=0
                s=0
                u=0
                t=0
            if(t==0):
                x=[q,r,s,u,0,0,0] 
            elif(t==1):
                x=[q,r,s,u,0,0,1] 
            elif(t==2):
                x=[q,r,s,u,0,1,0] 
            else:
                x=[q,r,s,u,1,0,0]
            x=np.array(x)
            x=x.reshape(1,-1)
            out=classifier_cue.predict(x)
            if out==1:
                cue_answer.append(i)
            else:
                cue_answer.append('_')
    else:
        cue_answer.append('@#$')
                          
i=0
while(i<len(lists)):
    count=0
    j=i
    while(j<len(lists)):
        if str(cue_answer[j]).isalpha():
            count=1
        if str(cue_answer[j])=='@#$':
               j=j+1
               break
        j=j+1
    if count==0:
        for k in range(i,j-1):
                        cue_answer[k] ='***'
    i=j
 
cue_answer[len(lists)-1]='***'            

i=0
scope_answer=[]
safty=scope_answer
while(i<len(lists)):
   if(cue_answer[i]=='***'):
           scope_answer.append('@')
           i=i+1
   elif(cue_answer[i]=='@#$'):
         scope_answer.append('@#$')
         i=i+1
   else:
        scope_answer.append(str(lists[4][i]))    
        i=i+1

i=0
while(i<len(lists)):
    if str(cue_answer[i])!='***' and str(cue_answer[i])!='@#$':
        sentence=""
        parser=""
        m=i
        scope=[]
        index=[]
        while(cue_answer[m]!='@#$'):
            if(str(cue_answer[m]).isalpha()):
                negative=cue_answer[m]
            sentence=sentence+scope_answer[m]+" "
            parser+=str(lists[6][m]).replace("*"," "+scope_answer[m])
            m=m+1
        tree=Tree.fromstring(parser)    
        traversal=tree.treepositions('leaves')
        word=tree.leaves()
        negative_position=word.index(negative)
        for j in range(0,len(word)):
         if(str(word[j]).isalpha()):
            if(j<negative_position):
                r_distance=0
                l_distance=negative_position-j
            elif(j>negative_position):
                l_distance=0
                r_distance=j-negative_position
            else:
                l_distance=0
                r_distance=0
            neg_tree_length=len(traversal[negative_position])
            neg_tree=traversal[negative_position]
            word_length=len(traversal[j])
            word_tree=traversal[j]
            if(neg_tree_length<=word_length):
                position=neg_tree_length
                for k in range(0,neg_tree_length):
                    if(neg_tree[k]!=word_tree[k]):
                        position=k
                        break
                distance=neg_tree_length-position
                distance+=word_length-position
            else:
                position=word_length
                for k in range(0,word_length):
                    if(neg_tree[k]!=word_tree[k]):
                        position=k
                        break
                distance=word_length-position
                distance+=neg_tree_length-position
            comma_count=0
            l=j
            if l<=negative_position:
                for p in range(l,negative_position):
                        if(str(word[p])==','):
                               comma_count+=1
            else:
                for p in range(negative_position,l):
                        if(str(word[p])==','):
                            comma_count+=1
            x=[l_distance,r_distance,distance,comma_count]
            x=np.array(x)
            x=x.reshape(1,-1)
            y=classifier_scope.predict(x)
            if(y==1):
                scope.append(word[j])
                flag3=0
        for j in range(i,m):
            if scope_answer[j] not in scope:
                    scope_answer[j]='_'
                       
        i=m
    else:
        i=i+1

for i in range(0,len(lists)):
    if(cue_answer[i].isalpha()):
        for j in cleaned_affix:
            flag1=0
            if(str(cue_answer[i]).find(j)!=-1):
                    flag1=1
                    a=cue_answer[i].split(j)
                    word_tag=tagger.parse(cue_answer[i])[0][2]
                    if(len(a[0])>1 or len(a[1])>1):
                        b=0
                        if(len(a[0])>=len(a[1])):
                            b=1
                            if wordnet.synsets(a[0]):
                                q=1
                            else:
                                q=0
                            ''' this is feature 3'''
                            m=tagger.parse(a[0])[0][2]
                            if m==word_tag:
                                r=1
                            else:
                                r=0
                            ''' this is  feature 4'''
                            antonyms = [] 
                            for syn in wordnet.synsets(cue_answer[i]):
                                for l in syn.lemmas():
                                    if l.antonyms(): 
                                        antonyms.append(l.antonyms()[0].name())    
                            s=0
                            for k in antonyms:
                                if(a[0].find(k)!=-1 or k.find(a[0])!=-1):
                                    s=1
                        else:
                            if wordnet.synsets(a[1]):
                                q=1
                            else:
                                q=0
                            if tagger.parse(a[1])[0][2]==word_tag:
                                r=1
                            else:
                                r=0
                            antonyms = [] 
                            for syn in wordnet.synsets(cue_answer[i]):
                                for l in syn.lemmas():
                                    if l.antonyms(): 
                                        antonyms.append(l.antonyms()[0].name())    
                            s=0
                            for k in antonyms:
                                if(a[1].find(k)!=-1 or k.find(a[1])!=-1):
                                    s=1
                        u=len(j)
                        if(len(a[0])==0 and len(a[1])>0):
                            t=1
                        elif(len(a[0])>0 and len(a[1])>0):
                            t=2
                        elif(len(a[0])>0 and len(a[1])==0):
                            t=3
                        else:
                            t=0
                        break
                    else:
                        q=0
                        r=0
                        s=0
                        u=0    
                        t=0
        if(flag1==0):
                q=0
                r=0
                s=0
                u=0
                t=0
        if(t==0):
                x=[q,r,s,u,0,0,0] 
        elif(t==1):
                x=[q,r,s,u,0,0,1] 
        elif(t==2):
                x=[q,r,s,u,0,1,0] 
        else:
                x=[q,r,s,u,1,0,0]
        x=np.array(x)
        x=x.reshape(1,-1)
        out=classifier_cue.predict(x)
        if out==1:
            if b==1:
                    scope_answer[i]=a[0]
                    cue_answer[i]=j
            else:
                    scope_answer[i]=a[1]
                    cue_answer[i]=j
                    
n=None                    
#Saving the result
final_sentences=[]                    
for i in range(len(lists)):
    sentence=""
    if( str(cue_answer[i])!='@#$'):
     if (str(cue_answer[i])!='***'):
        for j in range(0,7):
            sentence+=str(lists[j][i])+"	"
        sentence+=str(cue_answer[i])+"	"
        sentence+=str(scope_answer[i])+"	" 
        sentence+=str('_')
     else:
        for j in range(0,7):
            sentence+=str(lists[j][i])+"	"
        sentence+=str(cue_answer[i])+"	"
    else:
        sentence+=""
    final_sentences.append(sentence)

for i in final_sentences:
    text_file = open("test1_output.txt", "a")
    text_file.write("%s\n" % i)
    
#Test-2 file
with open('test2.txt') as f:
    lines = f.readlines()
    
list_words=[]
for i in range(0, len(lines)):
    list_words.append(lines[i].split())


    
lists=pd.DataFrame(list_words)

cue_answer=[]
for w in range(len(lists)):
    if(str(lists[0][w])!='None'):
        i=str(lists[4][w])
        if i.lower() in cleaned_negation:
            cue_answer.append(i)
        else:
            flag1=0
            for j in cleaned_affix:
                if (i.find(j)!=-1):
                    flag1=1
                    a=i.split(j)
                    word_tag=tagger.parse(i)[0][2]
                    if(len(a[0])>1 or len(a[1])>1):
                        if(len(a[0])>=len(a[1])):
                            if wordnet.synsets(a[0]):
                                q=1
                            else:
                                q=0
                            ''' this is feature 3'''
                            m=tagger.parse(a[0])[0][2]
                            if m==word_tag:
                                r=1
                            else:
                                r=0
                            ''' this is  feature 4'''
                            antonyms = [] 
                            for syn in wordnet.synsets(i):
                                for l in syn.lemmas():
                                    if l.antonyms(): 
                                        antonyms.append(l.antonyms()[0].name())    
                            s=0
                            for k in antonyms:
                                if(a[0].find(k)!=-1 or k.find(a[0])!=-1):
                                    s=1
                        else:
                            if wordnet.synsets(a[1]):
                                q=1
                            else:
                                q=0
                            if tagger.parse(a[1])[0][2]==word_tag:
                                r=1
                            else:
                                r=0
                            antonyms = [] 
                            for syn in wordnet.synsets(i):
                                for l in syn.lemmas():
                                    if l.antonyms(): 
                                        antonyms.append(l.antonyms()[0].name())    
                            s=0
                            for k in antonyms:
                                if(a[1].find(k)!=-1 or k.find(a[1])!=-1):
                                    s=1
                        u=len(j)
                        if(len(a[0])==0 and len(a[1])>0):
                            t=1
                        elif(len(a[0])>0 and len(a[1])>0):
                            t=2
                        elif(len(a[0])>0 and len(a[1])==0):
                            t=3
                        else:
                            t=0
                    else:
                        q=0
                        r=0
                        s=0
                        u=0    
                        t=0
            if(flag1==0):
                q=0
                r=0
                s=0
                u=0
                t=0
            if(t==0):
                x=[q,r,s,u,0,0,0] 
            elif(t==1):
                x=[q,r,s,u,0,0,1] 
            elif(t==2):
                x=[q,r,s,u,0,1,0] 
            else:
                x=[q,r,s,u,1,0,0]
            x=np.array(x)
            x=x.reshape(1,-1)
            out=classifier_cue.predict(x)
            if out==1:
                cue_answer.append(i)
            else:
                cue_answer.append('_')
    else:
        cue_answer.append('@#$')
                          
i=0
while(i<len(lists)):
    count=0
    j=i
    while(j<len(lists)):
        if str(cue_answer[j]).isalpha():
            count=1
        if str(cue_answer[j])=='@#$':
               j=j+1
               break
        j=j+1
    if count==0:
        for k in range(i,j-1):
                        cue_answer[k] ='***'
    i=j
 
cue_answer[len(lists)-1]='***'            

i=0
scope_answer=[]
safty=scope_answer
while(i<len(lists)):
   if(cue_answer[i]=='***'):
           scope_answer.append('@')
           i=i+1
   elif(cue_answer[i]=='@#$'):
         scope_answer.append('@#$')
         i=i+1
   else:
        scope_answer.append(str(lists[4][i]))    
        i=i+1

i=0
while(i<len(lists)):
    if str(cue_answer[i])!='***' and str(cue_answer[i])!='@#$':
        sentence=""
        parser=""
        m=i
        scope=[]
        index=[]
        while(cue_answer[m]!='@#$'):
            if(str(cue_answer[m]).isalpha()):
                negative=cue_answer[m]
            sentence=sentence+scope_answer[m]+" "
            parser+=str(lists[6][m]).replace("*"," "+scope_answer[m])
            m=m+1
        tree=Tree.fromstring(parser)    
        traversal=tree.treepositions('leaves')
        word=tree.leaves()
        negative_position=word.index(negative)
        for j in range(0,len(word)):
         if(str(word[j]).isalpha()):
            if(j<negative_position):
                r_distance=0
                l_distance=negative_position-j
            elif(j>negative_position):
                l_distance=0
                r_distance=j-negative_position
            else:
                l_distance=0
                r_distance=0
            neg_tree_length=len(traversal[negative_position])
            neg_tree=traversal[negative_position]
            word_length=len(traversal[j])
            word_tree=traversal[j]
            if(neg_tree_length<=word_length):
                position=neg_tree_length
                for k in range(0,neg_tree_length):
                    if(neg_tree[k]!=word_tree[k]):
                        position=k
                        break
                distance=neg_tree_length-position
                distance+=word_length-position
            else:
                position=word_length
                for k in range(0,word_length):
                    if(neg_tree[k]!=word_tree[k]):
                        position=k
                        break
                distance=word_length-position
                distance+=neg_tree_length-position
            comma_count=0
            l=j
            if l<=negative_position:
                for p in range(l,negative_position):
                        if(str(word[p])==','):
                               comma_count+=1
            else:
                for p in range(negative_position,l):
                        if(str(word[p])==','):
                            comma_count+=1
            x=[l_distance,r_distance,distance,comma_count]
            x=np.array(x)
            x=x.reshape(1,-1)
            y=classifier_scope.predict(x)
            if(y==1):
                scope.append(word[j])
                flag3=0
        for j in range(i,m):
            if scope_answer[j] not in scope:
                    scope_answer[j]='_'
                       
        i=m
    else:
        i=i+1

for i in range(0,len(lists)):
    if(cue_answer[i].isalpha()):
        for j in cleaned_affix:
            flag1=0
            if(str(cue_answer[i]).find(j)!=-1):
                    flag1=1
                    a=cue_answer[i].split(j)
                    word_tag=tagger.parse(cue_answer[i])[0][2]
                    if(len(a[0])>1 or len(a[1])>1):
                        b=0
                        if(len(a[0])>=len(a[1])):
                            b=1
                            if wordnet.synsets(a[0]):
                                q=1
                            else:
                                q=0
                            ''' this is feature 3'''
                            m=tagger.parse(a[0])[0][2]
                            if m==word_tag:
                                r=1
                            else:
                                r=0
                            ''' this is  feature 4'''
                            antonyms = [] 
                            for syn in wordnet.synsets(cue_answer[i]):
                                for l in syn.lemmas():
                                    if l.antonyms(): 
                                        antonyms.append(l.antonyms()[0].name())    
                            s=0
                            for k in antonyms:
                                if(a[0].find(k)!=-1 or k.find(a[0])!=-1):
                                    s=1
                        else:
                            if wordnet.synsets(a[1]):
                                q=1
                            else:
                                q=0
                            if tagger.parse(a[1])[0][2]==word_tag:
                                r=1
                            else:
                                r=0
                            antonyms = [] 
                            for syn in wordnet.synsets(cue_answer[i]):
                                for l in syn.lemmas():
                                    if l.antonyms(): 
                                        antonyms.append(l.antonyms()[0].name())    
                            s=0
                            for k in antonyms:
                                if(a[1].find(k)!=-1 or k.find(a[1])!=-1):
                                    s=1
                        u=len(j)
                        if(len(a[0])==0 and len(a[1])>0):
                            t=1
                        elif(len(a[0])>0 and len(a[1])>0):
                            t=2
                        elif(len(a[0])>0 and len(a[1])==0):
                            t=3
                        else:
                            t=0
                        break
                    else:
                        q=0
                        r=0
                        s=0
                        u=0    
                        t=0
        if(flag1==0):
                q=0
                r=0
                s=0
                u=0
                t=0
        if(t==0):
                x=[q,r,s,u,0,0,0] 
        elif(t==1):
                x=[q,r,s,u,0,0,1] 
        elif(t==2):
                x=[q,r,s,u,0,1,0] 
        else:
                x=[q,r,s,u,1,0,0]
        x=np.array(x)
        x=x.reshape(1,-1)
        out=classifier_cue.predict(x)
        if out==1:
            if b==1:
                    scope_answer[i]=a[0]
                    cue_answer[i]=j
            else:
                    scope_answer[i]=a[1]
                    cue_answer[i]=j
             
#Saving the result
final_sentences=[]   
               
for i in range(0,len(lists)-1):
    sentence=""  
    if( str(cue_answer[i])!='@#$'):
     if (str(cue_answer[i])!='***'):
        for j in range(0,7):
            sentence+=str(lists[j][i])+"	"
        sentence+=str(cue_answer[i])+"	"
        sentence+=str(scope_answer[i])+"	" 
        sentence+=str('_')
        final_sentences.append(sentence)        
     else:
        for j in range(0,7):
            sentence+=str(lists[j][i])+"	"
        sentence+=str(cue_answer[i])+"	"
        final_sentences.append(sentence)
    else:
        final_sentences.append("\n")
final_sentences.append("\n")

a=[]        
for i in range(0,len(final_sentences)):
    if final_sentences[i]=="\n":
        a.append(i)        

for i in range(0,len(final_sentences)):
    text_file = open("Test2_output.txt", "a")
    if final_sentences[i]!="\n":
        text_file.write("%s\n" % final_sentences[i])
    else:
        text_file.write("\n")


text_file = open("Test2_output.txt", "a")
text_file.write("\n")  
