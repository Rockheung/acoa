import sys, csv, os, sys
import imghdr
import socket
from urllib import urlretrieve
from multiprocessing import Pool as Thread
# from multiprocessing import Process

socket.setdefaulttimeout(30)

synset_list = open('synset_list.txt', 'r')
print 'synset_list roaded'

synset = map(lambda x: x[:9], synset_list.readlines())
print synset[0:5]

def filedown(filename, synset = synset):

    imgname, url = filename.split('\t')

    wordid = imgname.split('_')[0]

    target = os.path.join(rootpath, wordid)
    print target
    print synset[0:5]

#     if wordid in synset:
    if not os.path.exists(target):
        try:
            os.makedirs(target)
        except OSError:
            print 'Error: {}'.format(OSError)

    if not os.path.exists(os.path.join(target, imgname + '.jpg')):
        print 'Try to download' + imgname
        try :
            o = urlretrieve(url, os.path.join(target, imgname + '.jpg'))[0]
            if not imghdr.what(o) == 'jpeg':
                os.remove(o)
        except KeyboardInterrupt:
            return
        except :
            print imgname + 'Server has no response.'
            pass

rootpath = os.path.join(os.getcwd(), 'ImageNet_Fall_2011_Release')
if not os.path.exists( rootpath ):
    try:
        os.makedirs(rootpath)
    except OSError:
        print 'Error: {}'.format(OSError)


pool = Thread(128)

with open('fall11_urls_put_it_on.tsv', 'r') as f:
    pool.map(filedown, f)
    print 'filedown finished'

# pool.close()
# pool.join()

synset_list.close()
