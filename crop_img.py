import xml.etree.ElementTree as ET
import matplotlib as plt
import cv2
import argparse
import os
from multiprocessing import Pool as Thread
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--imgdir", help="target img path")
# parser.add_argument("--xmldir", help="target xml path")
args = parser.parse_args()

target = args.imgdir
print(target)
# target_xml = args.xmldir

def search(dirname, ext):
    result = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        if os.path.isdir(filepath):
            search(filepath, ext)
        else:
            if ext == os.path.splitext(filepath)[-1]:
                result.append(filepath)
    return result

def crop_img(imgpath):
    xmlpath = '.'.join(imgpath.split('.')[0:-1]) + '.xml'
    if not os.path.exists(xmlpath):
        print('XML file for\n\t' +  imgpath + '\nnot exists!')
        pass
    else : 
        cords = get_bbox_from_xml(xmlpath)
        img = cv2.imread(os.path.join(target,imgpath))
        i = 0
        for cord in cords:
            img_crop = img  [cord[0][1]
                            :cord[1][1]
                            ,cord[0][0]
                            :cord[1][0]]
            cv2.imwrite('.'.join(imgpath.split('.')[0:-1]) + '_cropped_' + str(i) + '.JPEG', img_crop)
            i += 1

def get_bbox_from_xml(file_name):
    doc = ET.parse(file_name)
    root = doc.getroot()
    cords = []
    for bndbox in root.findall('object/bndbox'):
        xmin = int(bndbox.findtext('xmin'))
        ymin = int(bndbox.findtext('ymin'))
        xmax = int(bndbox.findtext('xmax'))
        ymax = int(bndbox.findtext('ymax'))
        cords.append( ((xmin, ymin), (xmax, ymax)) )

    return cords

imglist = search(target, '.JPEG')
print(imglist)
pool = Thread(64)
pool.map(crop_img, imglist)
pool.close()
pool.join()

# files = os.listdir(target)
# bbox_target = os.path.join(target, 'bbox')
# if os.path.isdir(bbox_target) == False:
# 	os.mkdir(bbox_target)
# 
# 
# for image in files:
# 	if os.path.splitext(image)[1] == '.JPEG':
# 		name = os.path.splitext(image)[0]
# 		word_id, image_name = name.split('_')
# 		path = os.path.join('/home/acoa/ImageNet_Fall_2011_Release_all/ACOA data/total', word_id)
# 		xml_path = os.path.join(path, word_id+'_'+image_name+'.xml')
# 		points = get_bbox_from_xml(xml_path)
# 		img = cv2.imread(os.path.join(target,image))
# 		cv2.rectangle(img,points[0],points[1],(0,255,0),3)
# 		cv2.imwrite(os.path.join(bbox_target, image_name) + '.JPEG', img)
