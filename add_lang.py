ftype=['all','train','valid']
ltype=['_c.txt','_e.txt','_t.txt']

for fp in ftype:
    f0=open(fp+'.txt','wt')
    tlist=[]
    for lp in ltype:
        f1=open(fp+lp,'rt')
        flist=f1.readlines()
        f1.close()
        for any in flist:
            a=any.strip()
            tlist.append(a)
    for any in tlist:
        f0.write(any+'\n')
    f0.close()

