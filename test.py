import numpy as np

a = np.array([], np.int)
print a
b = np.ones((15,), np.int)
print b
a = np.concatenate((a, b), axis=None)
print a


# import re

# text = "feature_class_1.npy"
# m = re.match(r'feature_class_(\d+)\.npy', text)
# if m:
# 	print m.group(0), '\n', m.group(1)
# else:
# 	print 'not match'
