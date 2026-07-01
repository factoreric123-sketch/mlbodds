"""Score a PrizePicks strikeout slate with the validated K model."""
import unicodedata
import numpy as np
import pandas as pd
from scipy.stats import poisson
import kprops

def norm(n):
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(n.lower().split())

# (name, opp_team, line, is_home, goblin)
SLATE = [
    ("Trey Gibson","Chicago White Sox",4.0,True,False),
    ("Erick Fedde","Baltimore Orioles",2.5,False,True),
    ("Jacob deGrom","Cleveland Guardians",6.5,False,False),
    ("Tanner Bibee","Texas Rangers",5.0,True,False),
    ("Cristopher Sánchez","Pittsburgh Pirates",7.5,True,False),
    ("Bubba Chandler","Philadelphia Phillies",4.5,False,False),
    ("Cam Schlittler","Detroit Tigers",7.5,True,False),
    ("Tarik Skubal","New York Yankees",7.0,False,False),
    ("Kevin Gausman","New York Mets",5.5,True,False),
    ("Nolan McLean","Toronto Blue Jays",5.5,False,False),
    ("Connelly Early","Washington Nationals",5.5,True,False),
    ("Cade Cavalli","Boston Red Sox",4.5,False,False),
    ("Matthew Liberatore","Atlanta Braves",4.0,False,False),
    ("Martín Pérez","St. Louis Cardinals",4.0,True,False),
    ("Brandon Sproat","Cincinnati Reds",5.5,True,False),
    ("Rhett Lowder","Milwaukee Brewers",4.0,False,False),
    ("Griffin Jax","Kansas City Royals",5.0,False,False),
    ("Noah Cameron","Tampa Bay Rays",3.5,True,True),
    ("Matthew Boyd","San Diego Padres",5.0,True,False),
    ("JP Sears","Chicago Cubs",4.0,False,False),
    ("Joe Ryan","Houston Astros",7.0,False,False),
    ("Mike Burrows","Minnesota Twins",5.0,True,False),
    ("Eury Pérez","Colorado Rockies",5.0,False,False),
    ("Tanner Gordon","Miami Marlins",3.0,True,False),
    ("Bryan Woo","Los Angeles Angels",7.5,True,False),
    ("José Soriano","Seattle Mariners",5.5,False,False),
    ("Landen Roupp","Arizona Diamondbacks",5.0,False,False),
    ("Brandon Pfaadt","San Francisco Giants",3.0,True,False),
    ("Justin Wrobleski","Athletics",5.0,False,False),
    ("Jeffrey Springs","Los Angeles Dodgers",4.0,True,False),
]

df = pd.read_parquet(kprops.DATASET)
model = kprops.fit_model(df)
team_kr = kprops.team_krate(df)
lg = df["K"].sum()/df["bf"].sum()

g = df.sort_values(["pid","date","game_pk"]).groupby("name")
last = {}
for name, sub in g:
    last[norm(name)] = dict(
        krate=kprops.next_krate(sub["K"].to_numpy(float), sub["bf"].to_numpy(float)),
        bf_exp=sub["bf"].mean(), n=len(sub))

def p_over(mu, line): return float(1-poisson.cdf(np.floor(line), mu))

rows=[]
for name,opp,line,home,goblin in SLATE:
    k=norm(name)
    if k not in last:
        rows.append((name,line,goblin,None,None,None,None)); continue
    info=last[k]
    okr=team_kr.get(opp, lg)
    x=pd.DataFrame([[info["krate"],okr,info["bf_exp"],int(home)]],columns=kprops.FEATS)
    mu=float(model.predict(x)[0]); po=p_over(mu,line)
    rows.append((name,line,goblin,mu,po,info["n"],okr))

print(f"{'Pitcher':21}{'Line':>5}{'Gob':>4}{'muK':>6}{'P(Ov)':>7}{'n':>4}  pick(|edge|)")
print("-"*68)
picks=[]
for name,line,goblin,mu,po,n,okr in rows:
    if mu is None:
        print(f"{name:21}{line:5.1f}{'-':>4}{'--':>6}{'--':>7}{'--':>4}  not in universe"); continue
    side="MORE" if po>=0.5 else "LESS"; conf=abs(po-0.5)
    print(f"{name:21}{line:5.1f}{'G' if goblin else '-':>4}{mu:6.2f}{po:7.3f}{n:4d}  {side} ({conf:.3f})")
    picks.append((name,line,goblin,side,po,conf))
print("-"*68)
picks.sort(key=lambda t:-t[5])
print("\nStrongest leans:")
for name,line,goblin,side,po,conf in picks[:10]:
    print(f"  {side:4} {name:20}{line:5.1f}  P(over)={po:.3f}  conf={conf:.3f}{'  GOBLIN' if goblin else ''}")
