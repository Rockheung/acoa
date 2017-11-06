import sys, csv, os, sys
import imghdr
import socket
import csv
from urllib import urlretrieve
from multiprocessing import Pool as Thread

socket.setdefaulttimeout(30)

USERNAME = 'rockheung'
ACCESSKEY = '5c8b52496c9d8346fd9ff37240eaa4f0a421edab'
ROOT = os.path.join(os.getcwd(), 'ImageNet_Fall_2011_Release_all')

if not os.path.exists( ROOT ):
    try:
        os.makedirs(ROOT)
    except OSError:
        print 'Error: {}'.format(OSError)

if os.path.exists('words_syn.tsv'):
    os.remove('words_syn.tsv')


with open('api/words.tsv', 'r') as wd:
    with open('synset_list.txt', 'r') as f:
        syn2 = map(lambda x: x[:9], f.readlines())
        for line in wd:
            if line.split('\t')[0] in syn2:
                if os.path.exists(os.path.join(os.getcwd(), 'words_syn.tsv')):
                    with open('words_syn.tsv', 'a') as add:
                        add.write(line)
                else :
                    with open('words_syn.tsv', 'w') as new:
                        new.write(line)
    print 'words_syn.tsv has written.'

def tardown(wdid, ROOT = ROOT, user = USERNAME, key = ACCESSKEY):
    target = os.path.join(ROOT, wdid)
    wdmap_o = open('words_syn.tsv', 'r')
    wdmap = csv.reader(wdmap_o, delimiter='\t')

    for line in wdmap:
        nid, word = line[0], line[1]
        if nid == wdid:
            print 'Starting Download: ' + wdid
            url = 'http://www.image-net.org/download/synset?wnid=' + nid + '&username=' + user + '&accesskey=' + key + '&release=latest&src=stanford'
            o = urlretrieve(url, os.path.join(ROOT, nid + '_' + word  + '.tar'))[0]
            print o

pool = Thread(8)

# #     tardown(wdid = syn[0])
#     pool.map(tardown, syn)
#     print 'tardown finished'
#

with open('synset_list.txt', 'r') as f:
    syn = map(lambda x: x[:9], f.readlines())
    pool.map(tardown, syn)
    print 'tardown finished'

pool.close()
pool.join()

# synset_list.close()
# wdmap.close()
