import re
from time import gmtime, strftime
import numpy as np
import sklearn.datasets



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

pattern = re.compile(r"(\w*) (\w*) </s> \n<s>(\w*) (\w*)")
matches = pattern.finditer(substituted)
substituted = pattern.sub(r' ',substituted) # Finding words in radius 2

count = 0
for match in matches:
	print(match.group(1),"=>",match.group(2),"=>",match.group(3),"=>",match.group(4))
	count = count + 1

print(count)




# examples = []
# examples.append('some text')
# examples.append('another example text')
# examples.append('example 3')

# target = np.zeros((3,), dtype=np.int64)
# target[0] = 0
# target[1] = 1
# target[2] = 0
# dataset = sklearn.datasets.base.Bunch(data=examples, target=target)



f = open('2_output_'+strftime("%Y-%m-%d_%H:%M:%S", gmtime())
+'.txt','w')
f.write(substituted)
f.close()
