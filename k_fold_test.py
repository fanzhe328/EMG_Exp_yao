import numpy as np
from sklearn.cross_validation import KFold

def main():
	kf = KFold(279, n_folds=10)
	for train, test in kf:
		print len(train), len(test)


if __name__ == '__main__':
	main()