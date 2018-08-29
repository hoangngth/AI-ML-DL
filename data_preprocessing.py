# Data pre-processing

import os 

sentence = []
label = []

file = open(os.getcwd()+'/amazon_cells_labelled.txt', encoding='utf-8')
with file as f:
    content = f.readlines()
    for line in content:
        splitted = line.split('\t')
        print(splitted[0])
        sentence.append(splitted[0])
        label.append(splitted[1])
label = [x.strip('\n') for x in label]
label = list(map(int, label))

sentence_file = open(os.getcwd()+'/sentence.txt', 'w', encoding='utf-8')
for item in sentence:
    sentence_file.write("%s\n" % item)
label_file = open(os.getcwd()+'/label.txt', 'w', encoding='utf-8')
for item in label:
    label_file.write("%d\n" % item)