#author lmx 1124

import sys
import os

def get_elf_position(head):
	return (head + 60)

def get_size_position(head):
	return (head + 48)

def extra_o(filename):
	content = []
	with open(filename, 'rb') as a:
		content = a.read()

	#get head
	head = 0
	for i in range(len(content)):
		if content[i] == '\x2e' and content[i+1]=='\x6f' and content[i+2]=='\x2f' and content[i+3] == '\x20':
			head = i
			break
	for i in range(head-14, head):
		if ord(content[i])<37 or ord(content[i])>126:
			head = i+1

	print hex(head)

	#get size
	size_position = get_size_position(head)
	size = 0
	for i in range(size_position, size_position+8):
		if content[i]!='\x20':
			size = size *10 + (ord(content[i])-ord('\x30'))
		else:
			break

	#cycle extra
	while head < (len(content)-100):
		#get size
		size_position = get_size_position(head)
		size = 0
		for i in range(size_position, size_position+8):
			if content[i]!='\x20':
				size = size *10 + (ord(content[i])-ord('\x30'))
			else:
				break

		#get elf_name
		os.system("mkdir extra-"+filename)
		elf_name = './extra-'+filename+'/'
		for i in range(head, head+16):
			if ord(content[i])<33 or ord(content[i])>126 or ord(content[i]) ==47:
				break
			else:
				elf_name += content[i]
		print hex(head)
		if len(elf_name)==8:
			elf_name += 'default_name_'+str(size)
		print elf_name

		#get elf
		with open(elf_name, 'wb') as elf:
			for i in range(get_elf_position(head), get_elf_position(head)+size):
				elf.write(content[i])

		#re_position
		head = get_elf_position(head) + size


	
if __name__ == '__main__':
	#give name of .a
	if len(sys.argv)!=2:
		print "usage: python xxx.py xxx.a"
	else:
		extra_o(sys.argv[1])