import re
import numpy as np
import sklearn.ensemble
import xgboost
from random import shuffle
import copy

f = open('1.2_output_fullTest.txt','r')

#f = open('1.2_output_test.txt','r')
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


f = open('formatted_data.txt','w')

count_termi = 0
for match in matches_termi:
	f.write(match.group(1)+' '+match.group(2)+' '+'1'+'\n')
	count_termi = count_termi + 1

print("--------",count_termi)


count_no_termi = 0
for match_2 in matches_no_termi:
	f.write(match_2.group(1)+' '+'0'+'\n')
	count_no_termi = count_no_termi + 1
f.close()

print("--------",count_no_termi)



# feature_names = ["One", "Two", "Three", "Four"]
data = np.genfromtxt('formatted_data.txt', delimiter=' ', dtype=str)
shuffle(data)

# newrow = ['of','December','Please','pay','1']
# data = np.vstack([data, newrow])

print(data)

data_copy = copy.deepcopy(data)

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

# data = data.astype(float)


encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)
# np.random.seed(2)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.8,shuffle = False, stratify = None)
encoder.fit(data)
encoded_train = encoder.transform(train)

gbtree = xgboost.XGBClassifier(n_estimators=500, max_depth=50)
gbtree.fit(encoded_train, labels_train)
print(sklearn.metrics.accuracy_score(labels_test, gbtree.predict(encoder.transform(test))))

prediction = gbtree.predict(encoder.transform(test))
length = prediction.shape[0]
data_len = data.shape[0]
f = open('predictions.txt','w')
for i in range(length):
	#print(data_copy[data_len - i - 1], prediction[length - i - 1])
	f.write(str(data_copy[data_len - i - 1])+ str(prediction[length - i - 1])+'\n')
f.close()
# print(gbtree.predict(encoder.transform(test)))
f = open('for_understanding.txt','w')
f.write('---------------------------  data, labels -------------------------\n')
for i in range(train.shape[0]):
	f.write(str(train[i])+str(labels_train[i])+'\n')
for i in range(test.shape[0]):
	f.write(str(test[i])+str(labels_test[i])+'\n')
f.close()
