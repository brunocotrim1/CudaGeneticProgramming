import time
import numpy as np
import pandas as pd


df = pd.read_csv("data.csv")

funs = [ line.strip() for line in open("functions.txt").readlines() ]

def score(line):
    for u in ["sinf", "cosf", "tanf", "sqrtf", "expf"]:
        line = line.replace(u, f"np.{u[:-1]}")
    for c in df.columns:
        line = line.replace(f"_{c}_", f"(df[\"{c}\"].values)")
    a = eval(line)
    b = df["y"]
    squares = np.square(np.subtract(a, b))
    print(np.sum(squares))
    print(np.sum(squares)/len(squares))
    print(squares.shape)
    e = squares.mean()
    print(e)
    return e

l = funs[0]
funs=funs[:2]
print(score(l), l)
start = time.time()
r = min([ (score(line), line) for line in funs ])
end = time.time()
print(end - start)
print(f"{r[0]} {r[1]}")