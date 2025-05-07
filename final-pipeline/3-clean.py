import os
import re

unicode_lims = [b"\xe0\xae\x80", b"\xe0\xaf\xbf", b"\xe0\xa4\x80", b"\xe0\xa5\xbf"]

htmltag = r"<.*>"
replaceables = {
    "&quot;": '"',
    "&amp;": "&",
    "&gt;": ">",
    "&lt;": "<",
    "&#39;": "'",
    ".": " ",
    "?": " ",
    ",": " ",
    "!": " ",
    '"': "",
    "'": "",
    "(": " ",
    ")": " ",
    "&": "and",
    "-": "",
    ">": " ",
    "<": " ",
}
root = os.getcwd()
phrases = []
classified = []
# Setting the language to be aggregated

option = int(input("1.Tamil\n2.Hindi\n\nEnter: "))
mini = unicode_lims[option * 2 - 2]
maxi = unicode_lims[option * 2 - 1]

# Read from a single file instead of multiple files in a directory
input_file = os.path.join(root, "scraped.txt")

try:
    with open(input_file, "r", encoding="utf8") as file:
        unannotated = file.readlines()

        for comment in unannotated:
            cleanedcomment = re.sub(htmltag, " ", comment)
            for repla in replaceables.keys():
                cleanedcomment = cleanedcomment.replace(repla, replaceables[repla])
            words = cleanedcomment.split()
            newphrase = []
            for word in words:
                newword = ""
                for character in word:
                    coding = character.encode("utf-8")
                    if (
                        (coding < mini) or (coding > maxi)
                    ) and character.isascii() == False:
                        continue
                    newword = newword + character
                if newword != "":
                    newphrase.append(newword)
            if newphrase != []:
                phrases.append(newphrase)
except FileNotFoundError:
    print(f"Error: Could not find the file {input_file}")
    exit(1)

# Write the processed phrases to corpus.txt
with open("corpus.txt", "w", encoding="utf8") as file:
    for phrase in phrases:
        line = ""
        for word in phrase:
            line = line + " " + word
        file.write(line.strip() + "\n")

print(f"Processing complete. {len(phrases)} phrases written to corpus.txt")
