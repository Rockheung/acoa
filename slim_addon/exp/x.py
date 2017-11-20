networks_map = {'alexnet_v2': 2,
                'cifarnet': 3,
                'overfeat': 4,
                'addon' : 5,
                'vgg_a': 5,
}

name = 'addon'
if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)