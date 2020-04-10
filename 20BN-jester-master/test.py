import os
top = 'C:/Users/tranv/Downloads/jester-data/20bn-jester-v1'
al = []
for root, dirs, files in os.walk(top, topdown=False):
    for name in dirs:
        al.append(name)
        #print(os.path.join(name))

with open('newfile.txt', 'w') as f:
    for item in al:
        f.write("%s\n" % item)