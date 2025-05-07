import demoji
import json

demoji.download_codes()  # Load the latest emoticon dataset

l = []
with open("scraped.txt", "r", encoding="utf-8") as f:
    l = f.readlines()
emojis = {}  # Dict of line_no : emoji_name
for i in range(len(l)):
    a = l[i]
    emojidict = demoji.findall(a)
    if emojidict:
        emojis[i] = []
        for j in emojidict.values():
            j = j.replace(" ", "-")
            j = "<" + j + ">"
            emojis[i].append(j)

with open("emojis.json", "w") as f:
    json.dump(emojis, f)
