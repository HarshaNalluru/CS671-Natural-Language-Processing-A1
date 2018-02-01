from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian']

twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)

# print(twenty_train.target_names)
# print(len(twenty_train.data))
# print("\n".join(twenty_train.data[0].split("\n")[:3]))
# print(twenty_train.target_names[twenty_train.target[0]])
print(type(twenty_train))

# import numpy as np
# import sklearn.datasets

# examples = []
# examples.append('some text')
# examples.append('another example text')
# examples.append('example 3')

# target = np.zeros((3,), dtype=np.int64)
# target[0] = 0
# target[1] = 1
# target[2] = 0
# dataset = sklearn.datasets.base.Bunch(data=examples, target=target)
