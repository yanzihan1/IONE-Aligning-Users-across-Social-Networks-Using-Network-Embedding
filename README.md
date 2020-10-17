# IONE-Aligning-Users-across-Social-Networks-Using-Network-Embedding.
Aligning Users across Social Networks Using Network Embedding(IJCAI)，paper author uses Java,For wider application, we have updated the python version，use tensorflow1.14.0

Source Code and anonymous twitter_foursquare data for IJCAI 2016 paper "Aligning Users Across Social Networks Using Network Embedding"
Feel free to contact athuor (Liu Li liuli0407@hotmail.com) when you have any problems about the paper or the code(Java 8).



Feel free to contact me (Zihan Yan yzhcqupt@163.com) when you have any problems about the code(Python tensorflow1.14.0)

AcrossNetworkEmbeddingData

	foursquare:
	
		following: the relation file, "1  2" means user 1 is the follower of user 2.  			   			
		following.reverse: the reverse relation file, for model which considers only one direction context. ONE model.
		
	twitter:
	
		the same as the foursquare fold
		
	twitter_foursquare_groundtruth:
	
		groundtruth: the groundtruth for our experiment, the anchor users between twitter and foursquare. 
		Note that pls make the anchors as the *same* id during the pre-preparation, 
		although the testing anchors will have the same id, 
		they will *not* take part in the training progress as they are not contained in the groundtruth.x.foldtrain.train file.	
		
		groundtruth.x.foldtrain.train, the traning anchors, which are the 0.x of all the anchors.
		
		groundtruth.x.foldtrain.test,  the testing anchors, which are the 1-0.x of all the anchors.


#For the IONE, run the __main__.py.

#You should first run the __main__.py and second run the IONE_T.py to get the Twitter and Foursquare's embedding

We will add the IONES model with anonymous data as soon as possible.

All the embeddings in the embedding directory of foursquare and twitter.

The getPrecision.java is used for p@1-p@30 calculation of our model。 Note that in the evaluation, we use the |UnalignedUsers|=(|UnalignedAnchors|+|UnalignedNonAnchors|) as the candidate list. The Precison may be lower than metrics which only use |UnalignedAnchors| as the candidate list
