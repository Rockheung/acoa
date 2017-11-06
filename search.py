import os

def search(dirname, ext):
    result = []
    filenames = os.listdir(dirname)
#    print(filenames)
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        if os.path.isdir(filepath):
            search(filepath, ext)
        else:
            if ext == os.path.splitext(filepath)[-1]:
                result.append(filepath)
#                print result
    return result

path = '/home/rockheung/img_cropped/Dress/test'
print(search(path, '.JPEG'))
