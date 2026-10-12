import os

# file_in='relation2id.txt'
# file_out='readable_relations.txt'
file_in='entity2id.txt'
file_out='readable_entities.txt'



out=[]

with open(file_in,'r') as f:
	for line in f.readlines():
		# print(line)
		name,mid,_,_= line.strip().split()
		out.append([mid,name])

print(len(out))

with open(file_out,'w') as f:
	for line in out:
		f.write('\t'.join(line))
		f.write('\n')

print("Done")
