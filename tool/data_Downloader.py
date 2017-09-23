import sys, csv, os, sys
import imghdr
import socket
from urllib import urlretrieve
from multiprocessing.dummy import Pool as Thread

socket.setdefaulttimeout(30)

synset_list = open('synset_list.txt', 'r')
synset = synset_list.readlines()

def filedown(filename, synset_list = synset):

    imgname, url = filename.split('\t')

    wordid = imgname.split('_')[0]

    if wordid in synset_list:
        target = os.path.join(rootpath, wordid)
        if not os.path.exists(target):
            try:
                os.makedirs(target)
            except OSError:
                print 'Error: {}'.format(OSError)

        if not os.path.exists(os.path.join(target, imgname + '.jpg')):
            try :
                o = urlretrieve(url, os.path.join(target, imgname + '.jpg'))
                if not imghdr.what(o[0]) == 'jpeg':
                    os.remove(o[0])
            except :
                print 'Server has no response.'
                pass

rootpath = os.path.join(os.getcwd(), 'ImageNet_Fall_2011_Release')
if not os.path.exists( rootpath ):
    try:
        os.makedirs(rootpath)
    except OSError:
        print 'Error: {}'.format(OSError)


with open('fall11_urls.tsv', 'r') as f:
    pool = Thread(128)
    pool.map(filedown, f)
    pool.close()
    pool.join()

synset_list.close()

#     for line in f:
#         img, url = line.split('\t')
#         filedown(img, url)
#         print img
# #        print img
# #        print url
