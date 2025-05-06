import os
import csv

exit = 0


def annotator():
    if os.path.isfile("annotation.csv"):
        file = open("annotation.csv", "r", encoding="utf8")
        filecontents = csv.reader(file)
        anno_corpus = list(filecontents)
        file.close()
        return anno_corpus
    else:
        return [["comment", "labels"]]  # initializing the list


file = open("corpus.txt", "r", encoding="utf8")
lines = file.readlines()
j = (
    len(annotator()) - 1
)  # 0 if file doesn't exist, number of records already in the file if it does
for i in range(j, len(lines)):
    output = []
    cleanedline = lines[i].strip()
    words = cleanedline.split()
    for word in words:
        if not word.isascii():
            label = 0
            output.append(label)
        elif word.isdigit():
            label = 2
            output.append(label)
        else:
            print(
                f"\n1: Transliterated, 2: English/proper noun, 3: Error/NA, >4: exit. \n\n{word}\n"
            )
            label = 2
            if label == 1 or label == 2 or label == 3:
                output.append(label)
            else:
                print("\nExiting. Goodbye.")
                exit = 1
                break

    if exit == 1:
        break
    corpus = annotator()
    corpus.append([cleanedline, output])
    file = open("annotation.csv", "w", encoding="utf8", newline="")
    writer = csv.writer(file)
    writer.writerows(corpus)
    file.close()
