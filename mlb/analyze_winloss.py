import sys; sys.path.insert(0,"/Users/eric/Desktop/tennis bot/mlb")
import os; os.chdir("/Users/eric/Desktop/tennis bot/mlb")
import pandas as pd, glob, numpy as np, unicodedata

def norm(s):
    return "".join(c for c in unicodedata.normalize("NFKD",str(s)) if not unicodedata.combining(c)).lower().strip()

# all graded legs
rows=[]
for f in glob.glob('pp_logs/*/legs_*.csv'):
    df=pd.read_csv(f); df["name"]=df.pitcher.map(norm)
    rows.append(df)
g=pd.concat(rows,ignore_index=True)
g=g[g.leg_result.isin(['win','loss'])].copy()
g["won"]=(g.leg_result=='win').astype(int)

# attach book data where available (match date+name+stat)
books=[]
for f in glob.glob('pp_logs/books/books_2026-*.csv'):
    if 'detail' in f: continue
    b=pd.read_csv(f); books.append(b)
if books:
    bk=pd.concat(books,ignore_index=True)
    g=g.merge(bk[['date','name','stat','book_line','book_p_over']],on=['date','name','stat'],how='left')
else:
    g['book_line']=np.nan; g['book_p_over']=np.nan

# ---- candidate attributes ----
g["margin"]=g.actual - g.line                       # + = went over the line
g["hit_margin"]=np.where(g.side=='MORE', g.margin, -g.margin)  # + = leg won by this much
g["conf_bucket"]=pd.cut(g.p_hit,[0,.55,.60,.65,1],labels=['.50-.55','.55-.60','.60-.65','.65+'])
g["line_bucket"]=pd.cut(g.line,[0,3.5,4.5,5.5,99],labels=['<=3.5','4','5','5.5+'])
# does model side agree with book lean? (book_p_over>=.5 => book leans OVER)
def book_agree(r):
    if pd.isna(r.book_p_over): return 'no_book'
    book_side='MORE' if r.book_p_over>=.5 else 'LESS'
    return 'agree' if book_side==r.side else 'conflict'
g["book_agree"]=g.apply(book_agree,axis=1)
# is PP line softer than book? (MORE: PP line < book line good; LESS: PP line > book good)
def soft(r):
    if pd.isna(r.book_line): return 'no_book'
    if r.side=='MORE': return 'soft' if r.line<r.book_line else ('hard' if r.line>r.book_line else 'same')
    else: return 'soft' if r.line>r.book_line else ('hard' if r.line<r.book_line else 'same')
g["pp_line"]=g.apply(soft,axis=1)

def wr(df,by):
    t=df.groupby(by,observed=True).agg(n=('won','size'),wins=('won','sum'))
    t['win%']=(t.wins/t.n*100).round(0)
    t['pred%']=(df.groupby(by,observed=True).p_hit.mean()*100).round(0)
    return t

print(f"OVERALL: {g.won.sum()}/{len(g)} = {g.won.mean()*100:.0f}% win (pred {g.p_hit.mean()*100:.0f}%)\n")
for by,label in [('side','SIDE'),('stat','STAT'),('conf_bucket','MODEL CONFIDENCE'),
                 ('line_bucket','LINE HEIGHT'),('book_agree','MODEL vs BOOK SIDE'),('pp_line','PP LINE vs BOOK')]:
    print(f"=== by {label} ===")
    print(wr(g,by).to_string()); print()

# key contrast: winners vs losers avg attributes
print("=== WINNERS vs LOSERS: mean attribute values ===")
for col in ['p_hit','line','conf']:
    w=g[g.won==1][col].mean(); l=g[g.won==0][col].mean()
    print(f"  {col:8}: winners {w:.3f}  losers {l:.3f}  diff {w-l:+.3f}")
