# time 2018.11.11  24:00
# author lmx

# function : extract features from executable binary file
# pre_work [1.install ida_python 2.may generate or not]
from idautils import *
from idaapi import *
from idc import *
from idautils import DecodeInstruction
import logging
import networkx as nx
import sys, os, time
import json
IMM_MASK = 0xffffffff        #immediate num, mask
sys.path.append(os.getcwd()) #add current_path into search_path

# user defined op type?
OPTYPEOFFSET = 1000
o_string = o_imm + OPTYPEOFFSET
o_calls = OPTYPEOFFSET + 100 # Call 
o_trans = OPTYPEOFFSET + 101 # Transfer
o_arith = OPTYPEOFFSET + 102 # Arithmetic
transfer_instructions = ['MOV','PUSH','POP','XCHG','IN','OUT','XLAT','LEA','LDS','LES','LAHF', 'SAHF' ,'PUSHF', 'POPF']
arithmetic_instructions = ['ADD', 'SUB', 'MUL', 'DIV', 'XOR', 'INC','DEC', 'IMUL', 'IDIV', 'OR', 'NOT', 'SLL', 'SRL']


class CLogRecoder:

    def __init__(self, logfile = 'log.log', format = '%(asctime)s : %(message)s', level = logging.DEBUG):
        logging.basicConfig(filename= logfile, level= level , format= format)
        self._ft = format

    def addStreamHandler(self):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formater = logging.Formatter(self._ft)
        console.setFormatter(formater)
        logging.getLogger('').addHandler(console)
        return self

    def INFO(self, message):
        logging.info(message)
        return self

#log
ymd = time.strftime("%Y-%m-%d", time.localtime())
logger = CLogRecoder(logfile='./%s.log'% (ymd))
logger.addStreamHandler()


#file_cmd get file_bit_info
def get_ELF_bits(filename):
    import commands
    return 64


class Attributes_BlockLevel(object):
    def __init__(self, func_t):
        self._Blocks = set()             #basic block
        self._Blocks_list = []           #basic block list
        self._func = func_t              #init by func
        self._block_boundary = {}        #block boundary
        self._addr_func = func_t.startEA #enter_address of function
        self._name_func = str(GetFunctionName(func_t.startEA))  # GetFunctionName(startEA) returns the function name
        self._All_Calls = []             #get all call list
        self._G = nx.DiGraph()           #get the network_tp of func
        self._pre_nodes = {}             #pre_nodes of the block
        self._CFG = {}                   # key : Block startEA ; value : Block startEA of successors(hj_node)
        self._init_all_nodes()

        #compute betweenness
        self._Betweenness = nx.betweenness_centrality(self._G)  #compute the in_out of this block in graph

        #compute offspring by dfs[node]
        self._offspring = {}
        self.visit = set()
        for node in self._Blocks:
            self.visit = set()
            self._offspring[node] = self.dfs(node)

    # return node's offspring
    def dfs(self, node_startEA):
        if node_startEA in self.visit:            #have been visited, exit
            return 0
        self.visit.add(node_startEA)              #add start_node into visit     
        offspring = 0
        for succ_node in self._CFG[node_startEA]: #hj_node of current
            if succ_node not in self.visit:       #hj_node been visited
                offspring += self.dfs(succ_node) + 1
        return offspring

    # return node's offspring_list
    def dfs_list(self, node_startEA, dic):
        if node_startEA in self.visit:            #have been visited, exit
            return 0
        self.visit.add(node_startEA)              #add start_node into visit     
        for succ_node in self._CFG[node_startEA]: #hj_node of current
            if succ_node not in self.visit:       #hj_node been visited
                self.dfs(succ_node)
                dic.append(hex(succ_node))
        return dic

    # returns Oprand from instructions
    def _get_OpRands(self, ea):
        inst_t = DecodeInstruction(ea)
        return inst_t.Operands

    def _update_betweenness(self, added_node, pre):
        self._Betweenness[added_node] += 1
        if len(pre[added_node]) == 0:
            return
        queue = []
        queue+=pre[added_node]
        while len(queue) > 0:
            node = queue.pop(0)
            queue += pre[node]
            self._Betweenness[node] += 1

    def _add_predecessors(self, cbs, preb):
        for cb in cbs:
            if cb not in self._pre_nodes:
                self._pre_nodes[cb] = []
            self._pre_nodes[cb].append(preb)

    # initial block_boundary , get every node's range of address
    def _init_all_nodes(self):
        flowchart = FlowChart(self._func)
        for i in range(flowchart.size):
            basicblock = flowchart.__getitem__(i)
            self._Blocks.add(basicblock.startEA) #block list
            self._G.add_node(basicblock.startEA) #graph_tp
            self._CFG[basicblock.startEA] = [b.startEA for b in basicblock.succs()]
            for b in basicblock.succs():
                self._G.add_node(b.startEA)
                self._G.add_edge(basicblock.startEA, b.startEA)
            self._add_predecessors([b.startEA for b in basicblock.succs()], basicblock.startEA)
            self._block_boundary[basicblock.startEA] = basicblock.endEA
        self._Blocks_list = list(self._Blocks)
        self._Blocks_list.sort()

    #get pre_node_startEA
    def get_PreNodes_of_blocks(self, startEA):
        if startEA not in self._Blocks:
            return
        if startEA not in self._pre_nodes:
            return []
        return self._pre_nodes[startEA]

    # returns all Strings referenced in one block
    def get_All_Strings_of_Block(self, block_startEA):
        return self.get_OpValue_Block(block_startEA, my_op_type=o_string)

    # return a instruction's n'th oprand's reference
    def get_reference(self, ea, n):
        if (GetOpType(ea, n) == -1):
            return
        if (GetOpType(ea, n) == 1):
            print
            'General Register'
        if (GetOpType(ea, n) == 2):
            addr = GetOperandValue(ea, n)
            print
            'addr :', hex(Dword(addr))
            print
            ' reference'
            print
            'segment type :', GetSegmentAttr(addr, SEGATTR_TYPE)
            return GetString(Dword(addr))
        elif (GetOpType(ea, n) == 3):
            print
            'base + index'
        elif (GetOpType(ea, n) == 4):
            print
            'B+i+Displacement'
        elif (GetOpType(ea, n) == 5):
            print
            'immediate'
        elif (GetOpType(ea, n) == 6):
            print
            'far address'
        return GetOperandValue(ea, n)

    def get_AdjacencyMatrix(self):
        list = []
        for node in self._Blocks_list:
            newlist = []
            for node2 in self._Blocks_list:
                if node2 in self._CFG[node]:
                    newlist.append(1)
                else:
                    newlist.append(0)
            list.append(newlist)

        return list

    #offspring means children nodes in CFG
    def get_Offspring_of_Block(self, startEA):
        if startEA not in self._Blocks_list:
            return None
        return self._offspring[startEA]

    # there are some error to be solved
    # returns the next address of instruction which are in same basic block
    def get_next_instruction_addr(self, ea):
        return next(ea)

    # get_reference_data_one_block
    def get_reference_data_one_block(self, startEA):
        if (startEA not in self._block_boundary): # address is not right
            return
        endEA = self._block_boundary[startEA]
        it_code = func_item_iterator_t(self._func, startEA)
        ea = it_code.current()
        while (ea < endEA):
            yield (''.join(self.get_instruction(ea)))
            if (not it_code.next_code()):
                break
            ea = it_code.current()

    # get the whole instruction
    def get_instruction(self, ea):
        return idc.GetDisasm(ea)

    #get trans_instruction num
    def get_Trans_of_block(self, ea):
        return len(self.get_OpValue_Block(ea, o_trans))

    # startEA:basicblock's start address
    # return all instruction in one block, replaced by function get_reference_data_one_block
    def get_All_instr_in_one_block(self, startEA):
        return list(self.get_reference_data_one_block(startEA))

    # return function's name
    def getFuncName(self):
        return self._name_func

    # get full size of function frame
    def FrameSize(self):
        return GetFrameSize(self._func.startEA)  

    def getHexAddr(self, addr):
        return hex(addr)

    # get size of arguments in function frame which are purged upon return
    def FrameArgsSize(self): 
        return GetFrameArgsSize(self._func.startEA)

    def FrameRegsSize(self):  # get size of register_size
        return GetFrameRegsSize(self._func.startEA)

    # get operand value in one block
    def get_OpValue_Block(self, startEA, my_op_type):
        OPs = []
        if (startEA not in self._block_boundary):# address is not right
            return

        endEA = self._block_boundary[startEA]
        it_code = func_item_iterator_t(self._func, startEA)
        ea = it_code.current()
        while (ea < endEA):
            OPs += self.get_OpValue(ea, my_op_type)
            if (not it_code.next_code()):        # see if arrive end of the blocks
                break
            ea = it_code.current()

        return OPs

    def get_Arithmetics_Of_Block(self, ea):
        return len(self.get_OpValue_Block(ea, o_arith))

    # return all function or api names called by this function
    def get_Calls_BLock(self, ea):
        return len(self.get_OpValue_Block(ea, o_calls))

    # this is an abstract interface
    # it can replace functions like get_Numeric_Constant
    def get_OpValue(self, ea, my_op_type = o_void):
        OV = []
        if (my_op_type == o_trans): #it's a transfer instruction if data transfered between reg and mem
            inst = GetDisasm(ea).split(' ')[0].upper()
            if (inst in transfer_instructions):
                OV.append(inst)
            return OV

        elif( my_op_type == o_arith):
            inst = GetDisasm(ea).split(' ')[0].upper()
            if (inst in arithmetic_instructions):
                OV.append(inst)
            return OV

        op = 0
        op_type = GetOpType(ea, op)
        while (op_type != o_void):

            #o_calls
            if (my_op_type == o_calls):
                if (GetDisasm(ea).split(' ')[0].upper() == "CALL"):
                    OV.append(GetDisasm(ea).split(' ')[-1])
                    break

            if (op_type == my_op_type % OPTYPEOFFSET):
                ov = GetOperandValue(ea, op)
                ov &= 0xffffffff #only 32 bits
                if (my_op_type == o_imm):
                    logger.INFO(hex(ea) +' imm : ' + hex(ov))
                    if ov!=0:
                        OV.append(hex(ov))
                elif(my_op_type == o_string):
                    if (not SegName(ov) == '.rodata'):
                        addrx = list(DataRefsFrom(ov))
                        if len(addrx) == 0:
                            op += 1
                            op_type = GetOpType(ea, op)
                            continue
                        ov = addrx[0]
                    OV.append(GetString(ov))

            op += 1
            op_type = GetOpType(ea, op)
        return OV

    #get immediate num in blocks
    def get_Numeric_Constants_One_block(self, startEA):
        return self.get_OpValue_Block(startEA, my_op_type=o_imm)

    #get Betweenness of Blocks
    def get_Betweenness_of_Block(self, startEA):
        if startEA not in self._Betweenness:
            return -0
        return self._Betweenness[startEA]

    def get_CFG(self):
        return self._CFG

    # return all the start address of basicblock in form of set
    def get_All_Nodes_StartAddr(self):
        return self._Blocks_list

    # return a blocks end address
    def get_Block_Endaddr(self, startEA):
        if (startEA in self._block_boundary):
            return self._block_boundary[startEA]
        return -1

    #get func_diasm
    def get_func_diasm(self, startEA):
        instr = []
        func = ida_funcs.get_func(startEA)
        if not func:
            return instr
        fii = ida_funcs.func_item_iterator_t()
        instr_ok = fii.set(func)
        counter = 0
        while instr_ok:
            instr.append(idc.GetDisasm(fii.current()))
            instr_ok = fii.next_code()
            counter += 1
            if counter >=20:
                break
        return instr

    #get reg_param [rdi, rsi, rdx, rcx, r8, r9]
    def get_reg_param(self, startEA):
        reg_param = []

        instr = self.get_func_diasm(startEA)
        gz = '(add|sub|mul|div|xor|inc|dec|imul|idev|or|not|sll|srl|cmp)'
        para_di = 0
        para_si = 0
        para_dx = 0
        para_cx = 0
        para_r8 = 0
        para_r9 = 0
        # first use, get 1, first be_use, get-20
        for ins in instr:
            ins_element = ins.split(',')
            if len(ins_element):
                #len ==2 
                oper = re.findall(gz, ins_element[0])
                oper_type = len(oper)
                if len(ins_element) == 3:
                    #look [*],[],[]
                    if ('di' in ins_element[0]):
                        if para_di == 0:
                            para_di = -20
                    elif ('si' in ins_element[0]):
                        if para_si == 0:
                            para_si = -20
                    elif ('dx' in ins_element[0]):
                        if para_dx == 0:
                            para_dx = -20
                    elif ('cx' in ins_element[0]):
                        if para_cx == 0:
                            para_cx = -20
                    elif ('r8' in ins_element[0]):
                        if para_r8 == 0:
                            para_r8 = -20
                    elif ('r9' in ins_element[0]):
                        if para_r9 == 0:
                            para_r9 = -20
                    #look [],[*],[*]
                    if ('di' in ins_element[1]) or ('di' in ins_element[2]):
                        para_di += 1
                    elif ('si' in ins_element[1]) or ('si' in ins_element[2]):
                        para_si += 1
                    elif ('dx' in ins_element[1]) or ('dx' in ins_element[2]):
                        para_dx += 1
                    elif ('cx' in ins_element[1]) or ('cx' in ins_element[2]):
                        para_cx += 1
                    elif ('r8' in ins_element[1]) or ('r8' in ins_element[2]):
                        para_r8 += 1
                    elif ('r9' in ins_element[1]) or ('r9' in ins_element[2]):
                        para_r9 += 1

                elif len(ins_element) == 2:
                    #look [],[*]
                    if ('di' in ins_element[1]) or (('di' in ins_element[0]) and oper_type):
                        para_di += 1
                    elif ('si' in ins_element[1]) or (('si' in ins_element[0]) and oper_type):
                        para_si += 1
                    elif ('dx' in ins_element[1]) or (('dx' in ins_element[0]) and oper_type):
                        para_dx += 1
                    elif ('cx' in ins_element[1]) or (('cx' in ins_element[0]) and oper_type):
                        para_cx += 1
                    elif ('r8' in ins_element[1]) or (('r8' in ins_element[0]) and oper_type):
                        para_r8 += 1
                    elif ('r9' in ins_element[1]) or (('r9' in ins_element[0]) and oper_type):
                        para_r9 += 1
                    #look [*],[]
                    if ('di' in ins_element[0]) and not oper_type:
                        if para_di == 0:
                            para_di = -20
                    elif ('si' in ins_element[0]) and not oper_type:
                        if para_si == 0:
                            para_si = -20
                    elif ('dx' in ins_element[0]) and not oper_type:
                        if para_dx == 0:
                            para_dx = -20
                    elif ('cx' in ins_element[0]) and not oper_type:
                        if para_cx == 0:
                            para_cx = -20
                    elif ('r8' in ins_element[0]) and not oper_type:
                        if para_r8 == 0:
                            para_r8 = -20
                    elif ('r9' in ins_element[0]) and not oper_type:
                        if para_r9 == 0:
                            para_r9 = -20
                #len 1, mul edi
                elif len(ins_element) == 1:
                    #look [*]
                    if ('di' in ins_element[0]) and oper_type:
                        para_di += 1
                    elif ('si' in ins_element[0]) and oper_type:
                        para_si += 1
                    elif ('dx' in ins_element[0]) and oper_type:
                        para_dx += 1
                    elif ('cx' in ins_element[0]) and oper_type:
                        para_cx += 1
                    elif ('r8' in ins_element[0]) and oper_type:
                        para_r8 += 1
                    elif ('r9' in ins_element[0]) and oper_type:
                        para_r9 += 1
        if para_di > 0:
            reg_param.append('rdi')
        if para_si > 0:
            reg_param.append('rsi')
        if para_dx > 0:
            reg_param.append('rdx')
        if para_cx > 0:
            reg_param.append('rcx')
        if para_r8 > 0:
            reg_param.append('r8')
        if para_r9 > 0:
            reg_param.append('r9')
        '''reg_param.append(para_di)
        reg_param.append(para_si)
        reg_param.append(para_dx)
        reg_param.append(para_cx)
        reg_param.append(para_r8)
        reg_param.append(para_r9)'''
        return reg_param

#how to use this script
def print_help():
    help = 'script usage: argv[0]=self, argv[1]=json_file_name, [argv[2]=fun_name]\n'
    help = 'may args not enough'
    print(help)

# get block attributes
def get_att_block(blockEA, Attribute_Block):
    AB = Attribute_Block
    dic = []
    dic.append(len(AB.get_All_Strings_of_Block(blockEA)))
    dic.append(len(AB.get_Numeric_Constants_One_block(blockEA)))
    dic.append(AB.get_Trans_of_block(blockEA))
    dic.append(AB.get_Calls_BLock(blockEA))
    dic.append(len(AB.get_All_instr_in_one_block(blockEA)))
    dic.append(AB.get_Arithmetics_Of_Block(blockEA))
    dic.append(AB.get_Offspring_of_Block(blockEA))
    return dic

def save_Json(filename, func_name, binary_name):
    for i in range(0, get_func_qty()):
        fun = getn_func(i)
        segname = get_segm_name(fun.startEA)
        #check segment
        if 'text' not in segname:
            continue
        #name is null 
        if (func_name!= '' and GetFunctionName(fun.startEA) != func_name):
            continue
        with open(filename, 'a') as f:
            AB = Attributes_BlockLevel(fun)
            CFG = AB.get_CFG()
            #json write
            # recoder = {"src":"unknown","n_num": ,"succs": , "features": , "fname":}
            n_num = 0
            succs = []
            succ = []
            features = []
            fname = AB.getFuncName()

            #init succ_order
            counter = 0
            ea_set = set()
            ea_list = []
            for ea in CFG:
                if ea not in ea_set:
                    ea_set.add(ea)
                    ea_list.append(ea)
                succ_order = []
                for succ_ea in CFG[ea]:
                    if succ_ea not in ea_set:
                        ea_set.add(succ_ea)
                        ea_list.append(succ_ea)
                    c = 0
                    for element in ea_list:
                        if element == succ_ea:
                            succ_order.append(c)
                        c+=1
                CFG[ea] = succ_order
            #recoder block_7_dia_info
            for ea in CFG:
                n_num += 1
                for succ_order in CFG[ea]:
                    succ.append(succ_order)
                succs.append(succ)
                succ = []
                features.append(get_att_block(ea, AB))  #argv enter_address, class
            if n_num<=0:
                continue

            if get_ELF_bits(binary_name) ==32:
                frame = idc.GetFrame(fun.startEA)
                #no frame_staack space
                if frame is None:
                    continue
                ret_off = idc.GetMemberOffset(frame, ' r')
                first_arg_off = ret_off + 4
                #argv_size = frame_size - first_arg
                args_size = idc.GetStrucSize(frame) - first_arg_off
            else:
                reg_param = AB.get_reg_param(fun.startEA)
                reg_size = len(reg_param) * 8
                frame = idc.GetFrame(fun.startEA)
                
                if frame is None:
                    continue
                ret_off = idc.GetMemberOffset(frame, ' r')
                first_arg_off = ret_off + 4
                #argv_size = frame_size - first_arg
                args_size = idc.GetStrucSize(frame) - first_arg_off
                args_size += reg_size
            recoder = {"src":binary_name,"n_num":n_num ,"succs": succs, "features": features, "fname":fname, "param_size":args_size}
            json.dump(recoder, f, ensure_ascii=False)
            f.write('\n')
            
#script: /opt/ida-6.95/idal64 -S"Feature_Of_Binary.py sh_1.json sh" sh.i64
def main():
    if len(idc.ARGV) < 3:
        print_help()
        return
    func_name = '' 
    filename = idc.ARGV[1]
    binary_name = idc.ARGV[2]
    if len(idc.ARGV) >= 4:
        func_name = idc.ARGV[3]
    save_Json(filename, func_name, binary_name)


if __name__ == '__main__':
    main()
    idc.Exit()