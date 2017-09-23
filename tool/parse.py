import os
with open('synset_list.txt', 'r') as f:
    syn = map(lambda x: x[:9], f.readlines())

with open('fall11_urls.tsv', 'r') as g:
    for line in g:
        if line.split('\t')[0].split('_')[0] in syn:
            print 'found one' + line
            if os.path.exists(os.path.join(os.getcwd(), 'fall11_urls_put_it_on.tsv')):
                with open('fall11_urls_put_it_on.tsv', 'ajjj') as new:
                    new.write(line)
            else :
                with open('fall11_urls_put_it_on.tsv', 'w') as new:
                    new.write(line)

