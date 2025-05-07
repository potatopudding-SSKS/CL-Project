import random

l = []
with open("annotation.csv", "r", encoding="utf-8") as f:
    l = f.readlines()
ln = len(l)

s, nl = [], []
with open("train.csv", "w+", encoding="utf-8") as f:
    while len(s) < 8 * ln / 10:
        x = random.randrange(0, ln)
        if x not in s:
            f.write(l[x])
            nl.append(l[x])
            s.append(x)

for i in nl:
    l.remove(i)

with open("test.csv", "w+", encoding="utf-8") as f:
    f.writelines(l)
