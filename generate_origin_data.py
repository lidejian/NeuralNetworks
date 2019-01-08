import json
dict_sense_to_label = {
    'Temporal.Asynchronous.Precedence': 0,
    'Temporal.Asynchronous.Succession': 1,
    'Temporal.Synchrony': 2,
    'Contingency.Cause.Reason': 3,
    'Contingency.Cause.Result': 4,
    'Contingency.Condition': 5,
    'Comparison':6,
    'Comparison.Contrast': 6,
    'Comparison.Concession': 7,
    'Expansion':8,
    'Expansion.Conjunction': 8,
    'Expansion.Instantiation': 9,
    'Expansion.Restatement': 10,
    'Expansion.Alternative': 11,
    'Expansion.Alternative.Chosen alternative': 12,
    'Expansion.Exception': 13,
    'EntRel': 14,
}

with open('en_dev_relation_格式化.json',"r") as f,\
        open('sentence','w') as fsentence,\
        open('sence','w') as fsence,\
        open('label','w') as flabel:
    jrelation=json.load(f)
    docid=0
    for r in jrelation:
        if docid!=r['DocID']:
            if docid!=0:
                raw.close()
            raw =  open('./raw/%s'%r['DocID'],"r")
            print(r['DocID'])
            docid=r['DocID']
#         print(raw.read())
        arg1Span=r['Arg1']['CharacterSpanList']
        arg2Span=r['Arg2']['CharacterSpanList']
        sense=r['Sense'][0]
        ty=r['Type']
        
        start=arg1Span[0][0]
        if len(arg2Span)==1:
            end=arg2Span[0][1]
        else:
            end=arg2Span[1][1]
        print(r['ID'])
        print(arg1Span,end='--')
        print(arg2Span,start,end)
        
        
        raw.seek(start,0)
        sen=raw.read(end-start)
        if len(arg1Span)==2:# 如果有插入语，读入插入语
            raw.seek(arg1Span[0][1])
            parenthesis=raw.read(arg1Span[1][0]-arg1Span[0][1])
#             print('par:',parenthesis)
        if len(arg2Span)==2:# 如果有插入语
            raw.seek(arg2Span[0][1])
            parenthesis=raw.read(arg2Span[1][0]-arg2Span[0][1])
#             print('par:',parenthesis)
        sen = sen.replace(parenthesis,' ')#去除插入语
        sen = sen.replace('\n','')#去除\n
        
#         print(sen)
#         print(sense)
#         print(dict_sense_to_label[sense])
        fsentence.write(sen+'\n')
        fsence.write(sense+'\n')
        flabel.write(str(dict_sense_to_label[sense])+'\n')