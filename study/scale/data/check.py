"""python check.py FILE.json  -> lists banned-word hits in fields that must avoid them, count problems, duplicates."""
import json,re,sys
memes={m['id']:m for m in json.load(open('memes.json'))}
d=json.load(open(sys.argv[1])); bad=0
WANT={'explicit':5,'reworded':5,'pairs':8}
def stem(w): return re.sub(r"(ing|ed|es|s|'s)$","",w.lower())
def hits(t,ban):
    words=[stem(w) for w in re.findall(r"[A-Za-z']+",t)]; s=' '.join(words)
    return [b for b in ban if re.search(r'\b'+r'\s+'.join(map(re.escape,[stem(x) for x in re.findall(r"[A-Za-z']+",b)]))+r'\w*',s)]
for i in memes:
    if i not in d: print('MISSING',i); bad+=1; continue
    for k,v in d[i].items():
        if len(v)!=WANT[k]: print(i,k,'has',len(v),'want',WANT[k]); bad+=1
        if k=='reworded': chk=v
        elif k=='pairs': chk=[x for p in v for x in (p['present'],p['absent'])]
        else: continue
        for t in chk:
            h=hits(t,memes[i]['banned_words'])
            if h: print(i,k,h,'|',t); bad+=1
        if k=='pairs':
            for p in v:
                a,b=len(p['present'].split()),len(p['absent'].split())
                if abs(a-b)>4: print(i,'length gap',a,b,'|',p['present']); bad+=1
print('PROBLEMS',bad)
