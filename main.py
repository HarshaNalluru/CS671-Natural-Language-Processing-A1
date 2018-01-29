import re
from time import gmtime, strftime


sentence = "This hasn't been much that much of a twist and turn's to 'Tom','Harry' and u know who..yes its 'rock'"

f = open('test.txt','r')
message = f.read()
f.close()
#print(message)

#'(\w| )+[,!?]+' 

# pattern = re.compile(r"('(\w| )+[,!?]+' )|('(\w| )+[,!?\\n]+')")
# pattern = re.compile(r"(')((\w| )+[,!?.;\n]?)('|' )")
pattern = re.compile(r"(\n| )(')(([\w,?!;.\- ][\n]?)*)(')( |\n)")

# (')([\w,?!;\n']*)(')
matches = pattern.finditer(message)
# substituted = pattern.sub(r"\"\2\"",message)
# substituted = pattern.sub(r'"\2"',message)
substituted = pattern.sub(r'\1"\3"\6',message)

f = open('output_'+strftime("%Y-%m-%d_%H:%M:%S", gmtime())
+'.txt','w')
f.write(substituted)
f.close()

count = 0
for match in matches:
	print(match)
	count = count + 1

print(count)