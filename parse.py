from xml.etree.ElementTree import parse
tree = parse('/home/rockheung/img_cropped/Dress/test/n02728440_718.xml')
note = tree.getroot()
bndbox = note.find('object/bndbox')


