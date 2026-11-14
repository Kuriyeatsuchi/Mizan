"""Calibration: how often does each verdict come out when we know the truth?
Run with:  python calibration.py   (takes a few minutes)

AI assistance: drafted with Claude (Anthropic) during the hackathon. See README.md.
"""
import numpy as np, stats
Y=np.arange(1981,2026,dtype=float); n=len(Y)
def ar1(rng,phi,sd=1.0):
    e=rng.normal(0,sd,n); x=np.empty(n); x[0]=e[0]/np.sqrt(1-phi**2) if phi<1 else e[0]
    for i in range(1,n): x[i]=phi*x[i-1]+e[i]
    return x
def run(name,gen,reps=300):
    rng=np.random.default_rng(7); c={}; plain=0
    for _ in range(reps):
        x=gen(rng); l=stats.judge(Y,x).label; c[l]=c.get(l,0)+1
        plain+=stats.naive_p(Y,x)<stats.ALPHA   # what a spreadsheet trend line would report
    print(f"{name:42s}", {k:round(v/reps,3) for k,v in sorted(c.items())}, "plain line significant:", round(plain/reps,3))
run("noise only, independent", lambda r: ar1(r,0.0))
run("noise only, AR(1) 0.3", lambda r: ar1(r,0.3))
run("noise only, AR(1) 0.5", lambda r: ar1(r,0.5))
run("trend 0.3 sd/decade, AR 0.3", lambda r: 0.03*(Y-Y[0])+ar1(r,0.3))
run("trend 0.6 sd/decade, AR 0.3", lambda r: 0.06*(Y-Y[0])+ar1(r,0.3))
run("trend 1.0 sd/decade, AR 0.3", lambda r: 0.10*(Y-Y[0])+ar1(r,0.3))
run("step of 2 sd in 2001, AR 0.0", lambda r: ar1(r,0.0)+2.0*(Y>=2001))
run("step of 1 sd in 2001, AR 0.0", lambda r: ar1(r,0.0)+1.0*(Y>=2001))
