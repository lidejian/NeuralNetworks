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
        print(r['ID']-35707,'\t', r['ID'],end='\t')
        parenthesis='###hhh#####'
        midd='######hhh####'
        
        #读取start和end
        #start为arg1开始
        #end为arg2结束
        start=arg1Span[0][0]
        if len(arg2Span)==1:
            end=arg2Span[0][1]
        else:
            end=arg2Span[1][1]

        
        #读入num1和num2
        #num1为arg1结束
        #num2位arg2开始
        num1=arg1Span[0][1]#暂时记录arg1结尾位置
        num2=arg2Span[0][0]#记录arg2开头位置
        
        if len(arg1Span)==2:# 如果有插入语，读入插入语
            raw.seek(arg1Span[0][1],0)
            if arg1Span[1][0]-arg1Span[0][1] > 10:
                parenthesis=raw.read(arg1Span[1][0]-arg1Span[0][1])
            num1=arg1Span[1][1] #更新num1
#             print('par:',parenthesis)
        if len(arg2Span)==2:
            raw.seek(arg2Span[0][1],0)
            if arg2Span[1][0]-arg2Span[0][1] > 10:
                parenthesis=raw.read(arg2Span[1][0]-arg2Span[0][1])
            print('par:',arg2Span[0][1],arg2Span[1][0],parenthesis)
        
        #如果arg1和arg2位置反了
        if num2-num1<0:
            start,num1,num2,end=num2,end,start,num1
        
        
        #最后考虑连接词，更新start or end
        conn = r['Connective']['CharacterSpanList']
        if len(conn) ==1:
            if conn[0][0] < start:
                start = conn[0][0]
            if conn[0][1] > end:
                end=conn[0][1]
                
        print(start,num1,num2,end,end='\t')
        
        raw.seek(start,0)
        sen=raw.read(end-start)
        sen = sen.replace(parenthesis,' ')#去除插入语
        
        
        
        print(num2-num1,'\t',end-start,end='\t')
        
        #论元间相隔12个以上，认为不是连接词，去掉这部分。
        if num2-num1>12:
            raw.seek(num1,0)
            midd=raw.read(num2-num1)
            sen=sen.replace(midd,'.')
        
#         if num2-num1>12:
#             raw.seek(start,0)
#             sen1=raw.read(num1-start)
            
#             raw.seek(num2,0)
#             sen2=raw.read(end-num2)
            
#             sen3=sen1+'.'+sen2
            
#             sen3=sen3.replace(parenthesis,' ')
#             sen3=sen3.replace('\n','')
# #             print(sen)
# #             print('-'*20)
# #             print(sen3)
#             sen=sen3
            
        print(len(sen))
#         print(sen)
#         print(sense)
#         print(dict_sense_to_label[sense])

        sen = sen.replace('\n','')#去除\n
        fsentence.write(sen+'\n')
        fsence.write(sense+'\n')
        flabel.write(str(dict_sense_to_label[sense])+'\n')