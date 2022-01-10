f=open('T.txt')
fw=open('foursquare.embedding.update.2SameAnchor.1.foldtrain.twodirectionContext.number.100_dim.10000000','w')
for i in f:
    ii=i.split()
    strt=ii[0]+'_foursquare'+' '
    for iii in ii[1:]:
        strt=strt+iii+'|'
    fw.write(strt+'\n')

