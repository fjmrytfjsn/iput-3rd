from collections import defaultdict

hist1 = defaultdict(int)  # unigram (1-gram)
hist2 = defaultdict(int)  # bigram (2-gram)
# hist3=defaultdict(int) # trigram (3-gram)
with open("./aozora-small.txt", "r", encoding="utf-8") as f:  # open file in read mode
    for s in f:  # read line by line
        for i in range(0, len(s) - 3):  # iterate over each character in the line
            hist1[s[i : i + 1]] += 1  # count 1-gram
            hist2[s[i : i + 2]] += 1  # count 2-gram
            # hist3[s[i:i+3]] += 1 # count 3-gram
sorted_hist1 = sorted(
    hist1.items(), key=lambda x: x[1], reverse=True
)  # sort by frequency
prev = sorted_hist1[0][0]  # get the most frequent 1-gram
print(prev, end=" ")  # print the most frequent 1-gram as the first character
for i in range(20):  # repeat 20 times
    filtered_hist2 = {
        k: v for k, v in hist2.items() if k.startswith(prev)
    }  # filter 2-gram that starts with the prev character
    sorted_filtered_hist2 = sorted(
        filtered_hist2.items(), key=lambda x: x[1], reverse=True
    )  # sort by frequency
    prev = sorted_filtered_hist2[0][0][
        1
    ]  # get the most frequent 2-gram that start with the prev character,
    # and set it as the new prev character
    print(
        prev, end=" "
    )  # print second charactor of the most frequent 2-gram as the next character
print("")  # print new line
