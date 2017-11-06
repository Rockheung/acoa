import xml.etree.ElementTree as ET
import matplotlib as plt
import cv2
import argparse
import os
from matplotlib import pyplot as plt

def get_bbox_from_xml(file_name):
	doc = ET.parse(file_name)
	root = doc.getroot()
	bndbox = root.find('object/bndbox')
	xmin = int(bndbox.findtext('xmin'))
	ymin = int(bndbox.findtext('ymin'))
	xmax = int(bndbox.findtext('xmax'))
	ymax = int(bndbox.findtext('ymax'))
	return ((xmin, ymin), (xmax, ymax))


parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="folder path")
args = parser.parse_args()

folder = args.dir

files = os.listdir(folder)
bbox_folder = os.path.join(folder, 'bbox')
if os.path.isdir(bbox_folder) == False:
	os.mkdir(bbox_folder)


for image in files:
	if os.path.splitext(image)[1] == '.JPEG':
		name = os.path.splitext(image)[0]
		word_id, image_name = name.split('_')
		path = os.path.join('/home/acoa/ImageNet_Fall_2011_Release_all/ACOA data/total', word_id)
		xml_path = os.path.join(path, word_id+'_'+image_name+'.xml')
		points = get_bbox_from_xml(xml_path)
		img = cv2.imread(os.path.join(folder,image))
		cv2.rectangle(img,points[0],points[1],(0,255,0),3)
		cv2.imwrite(os.path.join(bbox_folder, image_name) + '.JPEG', img)
