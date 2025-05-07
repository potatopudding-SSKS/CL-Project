import csv

file = open("hindi/test.csv", "r",encoding="utf8")
filecontents = csv.reader(file)
anno_corpus = list(filecontents)

inpfile = open("hindi/pipeline-io/input.txt","w",encoding="utf8")
for line in anno_corpus:
    inpfile.write(line[0]+"\n")