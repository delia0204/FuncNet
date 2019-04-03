# time 2018.12
# author lmx
import os

#init1: only put binary ->sudo su
dir_name = './graph_data/'
b_name = os.listdir(dir_name)

#init2
os.system('mkdir vec_data')

dir_out = './vec_data/'

for binary in b_name:
	graph_name = dir_name + binary
	vec_name = dir_out + binary

	cmd = 'python graph_vec.py --fname '+ graph_name +' --rname ' + vec_name

	os.system(cmd)
	print 'over'