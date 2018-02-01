import re
from time import gmtime, strftime
import numpy as np
import sklearn.datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import lime
import lime.lime_tabular
import sklearn.ensemble
import xgboost
#sentence = "This hasn't been much that much of a twist and turn's to 'Tom','Harry' and u know who..yes its 'rock'"

# f = open('../test.txt','r')
# message = f.read()
# f.close()
#print(message)

#'(\w| )+[,!?]+' 

# pattern = re.compile(r"('(\w| )+[,!?]+' )|('(\w| )+[,!?\\n]+')")
# pattern = re.compile(r"(')((\w| )+[,!?.;\n]?)('|' )")
# pattern = re.compile(r"(\n| )(')(([\w,?!;.\-() ][\n]?)*)(')( |\n|;)")
# pattern = re.compile(r"(\n| )((Mr\.|\s\w\.\s|[\'\"\w,:;\-() ][\n]?)*)(\w\w\.|[!?])")

# (')([\w,?!;\n']*)(')
# matches = pattern.finditer(message)
# substituted = pattern.sub(r"\"\2\"",message)
# substituted = pattern.sub(r'"\2"',message)

# count = 0
# for match in matches:
# 	if count>10:
# 		break
# 	print()
# 	print("------------------")
# 	print("=>",match.group(1),"=>",match.group(2),"=>",match.group(3),"=>",match.group(4))
# 	print("------------------")
# 	print()
# 	count = count + 1
# print(count)
# substituted = pattern.sub(r'\1<s>\2\4</s>',message)

# pattern = re.compile(r"(\n| )(')(([\w,?!;.\-() \"\'][\n]?)*)(')( |\n)")
# final_substituted = pattern.sub(r'\1"\3"\6',substituted)


f = open('1.2_output.txt','r')
message = f.read()
f.close()

pattern = re.compile(r"[^<s>/\w ()]")

matches = pattern.finditer(message)

substituted = pattern.sub(r' ',message)

pattern = re.compile(r"<s>")
matches = pattern.finditer(substituted)
substituted = pattern.sub(r'\n<s>',substituted) # For readabity


pattern = re.compile(r"( )( )*")
matches = pattern.finditer(substituted)
substituted = pattern.sub(r' ',substituted) # Removing spaces

pattern = re.compile(r"(\w\w* \w\w*) </s> \n<s>(\w\w* \w\w*)")
matches_termi = pattern.finditer(substituted)
#Termi_dataset = pattern.sub(r' ',substituted) # Finding words in radius 2

pattern = re.compile(r"(\w\w* \w\w* \w\w* \w\w*)")
matches_no_termi = pattern.finditer(substituted)


examples = []
# examples.append('some text')
# examples.append('another example text')
# examples.append('example 3')
f = open('formatted_data.txt','w')
count_termi = 0
for match in matches_termi:
	# examples.append(match.group(1)+' '+match.group(2))
	f.write(match.group(1)+' '+match.group(2)+' '+'1'+'\n')
	# print("$$$$$$$$$$$$$$$")
	# print(match.group(1),"=>",match.group(2),"=>",match.group(3),"=>",match.group(4))
	count_termi = count_termi + 1

print("--------",count_termi)


count_no_termi = 0
for match in matches_no_termi:
	f.write(match.group(1)+' '+'0'+'\n')
	# examples.append(match.group(1))
	count_no_termi = count_no_termi + 1
f.close()

print("--------",count_no_termi)

target = np.zeros((count_no_termi + count_termi,), dtype=np.int64)
for i in range(count_termi):
	target[i] = 1

#print(examples)


# feature_names = ["One", "Two", "Three", "Four"]
data = np.genfromtxt('formatted_data.txt', delimiter=' ', dtype=str)
print(data)
labels = data[:,4]
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
# class_names = le.classes_
data = data[:,:-1]
categorical_features = [0,1,2,3]

categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_

data = data.astype(float)
encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)
np.random.seed(1)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)
encoder.fit(data)
encoded_train = encoder.transform(train)

gbtree = xgboost.XGBClassifier(n_estimators=500, max_depth=50)
gbtree.fit(encoded_train, labels_train)
print(sklearn.metrics.accuracy_score(labels_test, gbtree.predict(encoder.transform(test))))


f = open('for_understanding.txt','w')
f.write('---------------------------  data, labels -------------------------\n')
for i in range(train.shape[0]):
	f.write(str(train[i])+str(labels_train[i])+'\n')
for i in range(test.shape[0]):
	f.write(str(test[i])+str(labels_test[i])+'\n')

# f.write('------------------------------------------------------------')
# f.write('---------------------------  test -------------------------')
# f.write(test)
# f.write('------------------------------------------------------------')
# f.write('---------------------------  labels_train -------------------------')
# f.write(labels_train)
# f.write('------------------------------------------------------------')
# f.write('---------------------------  labels_test -------------------------')
# f.write(labels_test)
# f.write('------------------------------------------------------------')
f.close()


# predict_fn = lambda x: gbtree.predict_proba(encoder.transform(x)).astype(float)
# explainer = lime.lime_tabular.LimeTabularExplainer(train ,feature_names = feature_names,class_names=class_names,
#                                                    categorical_features=categorical_features, 
#                                                    categorical_names=categorical_names, kernel_width=3)


# target = np.zeros((3,), dtype=np.int64)
# target[0] = 0
# target[1] = 1
# # target[2] = 0
# dataset = sklearn.datasets.base.Bunch(data=examples, target=target)

# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(dataset.data)
# print(X_train_counts.shape)

# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# clf = MultinomialNB().fit(X_train_tfidf, dataset.target)

# docs_new = ['God is Love you', 'OpenGL on the GPU', 'were heard Not all']
# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predicted = clf.predict(X_new_tfidf)

# print(predicted)

# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, dataset.target_names[category]))



# f = open('2_output_'+strftime("%Y-%m-%d_%H:%M:%S", gmtime())
# +'.txt','w')
# f.write(substituted)
# f.close()
