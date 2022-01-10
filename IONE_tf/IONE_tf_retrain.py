from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from graph import *
from src.openne.classify import Classifier, read_node_label
from src.openne import double_up
import time


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', default='../../data/IONE/T.edge',   #twiiter.edge
                        help='Input graph file')
    parser.add_argument('--output',default='twitter.txt',          #twitter的embedding
                        help='Output representation file')
    parser.add_argument('--train_anchor_path', default='../../data/IONE/train',
                        help='Output representation file')

    #==========================================================================================
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
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--q', default=1.0, type=float)
    parser.add_argument('--method', default='line', help='The learning method')
    parser.add_argument('--graph-format', default='edgelist',
                        help='Input graph format')
    parser.add_argument('--negative-ratio', default=5, type=int,
                        help='the negative ratio of LINE')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--no-auto-save', action='store_true',
                        help='no save the best embeddings when training LINE')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--representation_size', default=100, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--hidden', default=16, type=int,
                        help='Number of units in hidden layer 1')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--network', default='F',
                        help='social network')
    parser.add_argument('--label_file', default='',
                        help='social network label')
    parser.add_argument('--encoder-list', default='[1000, 128]', type=str,
                        help='a list of numbers of the neuron at each encoder layer, the last number is the '
                             'dimension of the output node representation')
    args = parser.parse_args()

    if not args.output:
        print("No output filename. Exit.")
        exit(1)
    return args

def main(args,train_output_file):
    t1 = time.time()
    g = Graph()
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input,
                        directed=args.directed)

    if args.label_file and not args.no_auto_save:
        model = double_up.IONE(train_output_file,g, epoch=args.epochs, rep_size=args.representation_size, order=args.order,label_file=args.label_file, clf_ratio=args.clf_ratio,train_file=args.train_anchor_path)
    else:
        model = double_up.IONE(train_output_file,g, epoch=args.epochs,rep_size=args.representation_size, order=args.order,train_file=args.train_anchor_path)
    t2 = time.time()
    model.save_embeddings(args.output)
    if args.label_file and args.method != 'gcn':
        vectors = model.vectors
        X, Y = read_node_label(args.label_file)
        print("Training classifier using {:.2f}% nodes...".format(
            args.clf_ratio*100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        clf.split_train_evaluate(X, Y, args.clf_ratio, seed=0)

def retrain(train_output_file):
    random.seed(123)
    np.random.seed(123)
    main(parse_args(),train_output_file)

