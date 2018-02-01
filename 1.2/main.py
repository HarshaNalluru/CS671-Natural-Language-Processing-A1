import re
from time import gmtime, strftime


#sentence = "This hasn't been much that much of a twist and turn's to 'Tom','Harry' and u know who..yes its 'rock'"

f = open('../test.txt','r')
# f = open('../fullTest.txt','r')
message = f.read()
f.close()
#print(message)

#'(\w| )+[,!?]+' 

# pattern = re.compile(r"('(\w| )+[,!?]+' )|('(\w| )+[,!?\\n]+')")
# pattern = re.compile(r"(')((\w| )+[,!?.;\n]?)('|' )")
# pattern = re.compile(r"(\n| )(')(([\w,?!;.\-() ][\n]?)*)(')( |\n|;)")
pattern = re.compile(r"(\n| )((Mr\.|\s\w\.\s|[\'\"\w,:\-() ][\n]?)*)(\w\w\.|[!?]|--|;)")

# (')([\w,?!;\n']*)(')
matches = pattern.finditer(message)
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
substituted = pattern.sub(r'\1<s>\2\4</s>',message)

# pattern = re.compile(r"(\n| )(')(([\w,?!;.\-() \"\'][\n]?)*)(')( |\n)")
# final_substituted = pattern.sub(r'\1"\3"\6',substituted)


f = open('1.2_output_'+strftime("%Y-%m-%d_%H:%M:%S", gmtime())
+'.txt','w')
f.write(substituted)
f.close()

count = 0
for match in matches:
	#print(match)
	count = count + 1

print(count)