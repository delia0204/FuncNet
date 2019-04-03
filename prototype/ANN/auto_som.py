import os

dir_name = './vector/'
result_name = './map/'
vector_name = os.listdir(dir_name)
os.system('mkdir map')

for vectors in vector_name:
	vector_file = dir_name + vectors
	map_file = result_name + vectors
	cmd = 'python map.py ' +vector_file+ ' ' +map_file + '\n'
	os.system(cmd)