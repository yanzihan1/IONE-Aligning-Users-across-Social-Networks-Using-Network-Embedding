# IONE-Aligning-Users-across-Social-Networks-Using-Network-Embedding.
Aligning Users across Social Networks Using Network Embedding(IJCAI)，paper author uses Java(https://github.com/ColaLL/IONE),For wider application, we have updated the python version，use tensorflow1.14.0

Source Code and anonymous twitter_foursquare data for IJCAI 2016 paper "Aligning Users Across Social Networks Using Network Embedding"
Feel free to contact athuor (Liu Li liuli0407@hotmail.com) when you have any problems about the paper or the code(Java 8).



Feel free to contact me (Zihan Yan yzhcqupt@163.com) when you have any problems about the code(Python tensorflow1.14.0)




#For the IONE, run the __main__.py.

#You should first run the __main__.py and second run the IONE_T.py to get the Twitter and Foursquare's embedding

```
T.edge mains twitter graph

F.edge mains fousquare graph

train file mains anchor set,You can change train ratio by changing the 'p',such as p=0.1 represents train ratio = 10%

If you want to use another social network, pls modify the parameter size of the network,Here is twitter(5120),foursquare(5313).
```

Experimental comparison：(We have done a number of experiments and take the average value such as train ratio10%, p@1,p@5,p@10,p@15,p@20,p@30)

IONE(JAVA):0.0149,0.0534,0.0929,0.09548,0.129,0.158
IONE(python):#0.0131,0.0527,,0.0999,111,0.137,0.166

Accuracy in other experiments：-0.5 ~ +0.7


