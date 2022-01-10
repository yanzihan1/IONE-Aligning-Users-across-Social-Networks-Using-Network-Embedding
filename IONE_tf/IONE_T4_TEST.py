
anchor_emd_file='T.txt'
train_file='train'
class Data_mask(object):
    def __init__(self, train_file,anchor_emd_file,nodes_number,dim):
        f_train = open(train_file)
        self.dim=dim
        self.anchor_emd_file=anchor_emd_file
        self.nodes_number=nodes_number
        self.train = []
        for node in f_train:
            self.train.append(int(node))
        self.Mask_mat = [[1.] * int(self.dim / 2)] *self.nodes_number
        self.anchor_emd = [[0.] * int(self.dim / 2)] * self.nodes_number

    def mask(self,dic):
        f=open(self.anchor_emd_file)
        for train_node in self.train:
            t_node=dic.get(str(train_node))
            self.Mask_mat[t_node]=[0.]*int(self.dim/2)
        return self.Mask_mat
    def anc_emd(self,dic,order):
        if order==1:
            f = open(anchor_emd_file)
            ff = f.readlines()
            for line in ff[1:]:
                lines=line.split()
                if int(lines[0]) in self.train:
                    t_node=dic.get(lines[0])
                    emd=lines[1:]
                    emd=[float(x) for x in emd]
                    self.anchor_emd[t_node] = []
                    for i in range(0,int(self.dim/2)):
                        self.anchor_emd[t_node].append(float(emd[i]))
            return self.anchor_emd
        elif order == 2:
            f = open(anchor_emd_file)
            ff = f.readlines()
            for line in ff[1:]:
                lines = line.split()
                if int(lines[0]) in self.train:
                    t_node = dic.get(lines[0])
                    emd = lines[1:]
                    emd = [float(x) for x in emd]
                    self.anchor_emd[t_node] = []
                    for i in range(int(self.dim/2), self.dim):
                        self.anchor_emd[t_node].append(float(emd[i]))
            return self.anchor_emd



