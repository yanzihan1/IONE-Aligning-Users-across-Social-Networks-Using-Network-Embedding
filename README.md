# IONE-Aligning-Users-across-Social-Networks-Using-Network-Embedding
Aligning Users across Social Networks Using Network Embedding(IJCAI)，paper author uses Java(https://github.com/ColaLL/IONE)   
### For wider application, we have updated the python version，use tensorflow1.14.0  

Source Code and anonymous twitter_foursquare data for IJCAI 2016 paper "Aligning Users Across Social Networks Using Network Embedding"    

### Welcom to contact me (Zihan Yan yzhcqupt@163.com) when you have any problems about the code(Python tensorflow1.14.0)  


You just need run 'IONE_tf_train.py'    

In order to be fair (actually lazy), we convert the output embedding into an embedding file in ione java version format. You can test the accuracy in java files.run 'emd_to_ione_emd.py'  


Before, some students talked privately about the code I wanted tensorflow. Because I'm a little busy, the code didn't come up. So apologize!  

The code style may not be very good, Sry!   
### requirements    
numpy==1.14  
networkx==2.0  
scipy==0.19.1  
tensorflow>=1.12.1  
gensim==3.0.1  
scikit-learn==0.19.0  
### test result：  
   1- The tensorflow version of ione because using GPU runs faster than the Java version.  
   2- The tensorflow version of ione is almost as accurate as the Java version. Sometimes it is better. It may be because of the use of the deep learning optimizer.  
   
### The code  refers to the open source tool 'OpenNE' of Tsinghua University.  OpenNE: An open source toolkit for Network Embedding           https://github.com/thunlp/OpenNE  


   
   






