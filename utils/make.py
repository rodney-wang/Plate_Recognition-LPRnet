#coding=utf-8
import json

ff=open('carplate.json', 'w')

for i, c in enumerate(open('./chars').readline().strip().decode('utf-8')):
    print i, c


char_dict = {c:i for i, c in enumerate(open('./chars').readline().strip().decode('utf-8'))}


json.dump(char_dict, ff, sort_keys=True)
