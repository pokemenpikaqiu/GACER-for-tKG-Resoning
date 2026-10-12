import os

valid_output_file='./intervals/valid.txt'
test_output_file='./intervals/test.txt'

def extract(input_file, output_file):
	out=[]
	skip_count=0
	with open(input_file,'r') as f:
		lines=f.readlines()
		for l in lines:
			a=l.strip().split()
			if(len(a)==5):
				start=a[-2].split('-')[0]
				end=a[-1].split('-')[0]

				if(start=="####" or end=="####"): #if either is missing, skip
					skip_count+=1
					continue
				out.append(l)


	with open(output_file,'w') as f:
		for l in out:
			f.write(l.strip())
			f.write('\n')

	print("Saved filtered valid to {}, new length={}, skipped {}".format(output_file, len(out), skip_count))


# print("Saved filtered valid to {}".format(test_output_file))


if __name__=='__main__':
	extract('valid.txt', valid_output_file)
	extract('test.txt', test_output_file)
