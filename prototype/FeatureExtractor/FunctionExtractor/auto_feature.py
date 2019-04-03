# time 2018.11.11  24:00
# author lmx

import os

#init1: only put binary ->sudo su
dir_name = './binary/'
b_name = os.listdir(dir_name)

#init2
os.system('mkdir binary_feature')

for files in b_name:
	dir_name = './binary/'+files+'/'
	binarys = os.listdir(dir_name)
	feature_json = './binary_feature/'+files+'.json'

	for binary in binarys:
		b_path_name = './binary/'+files+'/'+binary
		cmd = '/opt/ida-6.95/idal64 -A ' + b_path_name + '\n'
		cmd += '/opt/ida-6.95/idal64 -A -S"Feature_Of_Binary.py '+ feature_json +' '+ files+binary +'" '+ b_path_name[:-2]+ '.i64' +'\n'
		cmd += 'rm ' + b_path_name[:-2]+ '.i64' + '\n'
		os.system(cmd)
		print 'over'