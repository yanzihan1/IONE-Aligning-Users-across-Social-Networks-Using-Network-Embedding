from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from graph import *
from src.openne.classify import Classifier, read_node_label
from src.openne import IONE_update
from src.openne.IONE_tf_retrain import retrain

import time


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', default='../../data/IONE/F.edge',    #这是边的文件可以设置foursquare或者twitter，这里用的foursquare
                        help='Input graph file')
    parser.add_argument('--output',default='IONE_f_embeddding.txt',     #输出地址,foursquare的embedding
                        help='Output representation file')
    '''
         下列anchor文件的格式保存为txt就行,里面格式:
          1
          2
          3
          4
          5                 
    '''
    parser.add_argument('--train_anchor_path', default='../../data/IONE/train',
                        help='Output representation file')
#==============================================setting==============================================
    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--representation-size', default=100, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The training epochs of LINE and GCN')
    parser.add_argument('--method', default='line', help='The learning method')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    parser.add_argument('--graph-format', default='edgelist',
                        help='Input graph format')
    parser.add_argument('--negative-ratio', default=5, type=int,
                        help='negative ratio')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose TRAIN WAY')
    parser.add_argument('--no-auto-save', action='store_true',
                        help='no save the best embeddings when training')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--network', default='F',
                        help='social network')
    args = parser.parse_args()
    return args


def main(args):
    t1 = time.time()
    g = Graph()
    print("Reading...")
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input,
                        directed=args.directed)
    if args.label_file and not args.no_auto_save:
        model = IONE_update.IONE(g, epoch=args.epochs, rep_size=args.representation_size, order=args.order,
                          label_file=args.label_file, clf_ratio=args.clf_ratio,train_file_path=args.train_anchor_path)
    else:
        model = IONE_update.IONE(g, epoch=args.epochs,
                          rep_size=args.representation_size, order=args.order,train_file_path=args.train_anchor_path)
    t2 = time.time()
    print("Saving embeddings...")
    model.save_embeddings(args.output)
    if args.label_file:
        vectors = model.vectors
        X, Y = read_node_label(args.label_file)
        print("Training classifier using {:.2f}% nodes...".format(
            args.clf_ratio*100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        clf.split_train_evaluate(X, Y, args.clf_ratio, seed=0)

if __name__ == "__main__":
    random.seed(123) #IONE java代码中保留的random seed 123
    np.random.seed(123)
    main(parse_args())
    args=parse_args()
    retrain(args.output)
