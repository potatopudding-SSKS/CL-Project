import os
import csv
import msvcrt

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
        texit = 0
        while not texit:
            if not word.isascii():
                label = 0
                output.append(label)
                texit = 1
            elif word.isdigit():
                label = 2
                output.append(label)
                texit = 1
            else:
                print(word)
                try:
                    # Wait for a keypress (blocking)
                    key_byte = msvcrt.getch()
                    # Convert the byte to string (e.g., b'1' to '1')
                    key_str = key_byte.decode("utf-8")
                    print(f"Key pressed: {key_str}")
                    # Check if the key is a digit
                    if key_str.isdigit():
                        label = int(key_str)
                        if label in [1, 2, 3]:
                            output.append(label)
                            texit = 1
                        elif label == 0:
                            print("\nExiting. Goodbye.")
                            exit = 1
                            texit = 1
                            break
                        else:
                            print("\nIncorrect Tag, try again")
                except Exception:
                    print("\nPlease try again")
        if exit:
            break

    if exit == 1:
        break
    corpus = annotator()
    corpus.append([cleanedline, output])
    file = open("annotation.csv", "w", encoding="utf8", newline="")
    writer = csv.writer(file)
    writer.writerows(corpus)
    file.close()
