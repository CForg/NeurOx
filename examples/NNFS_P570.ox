/* This replicates the code on Page 344 of NNFS usng NeurOx */
#import "NeurOx"        //import the NeurOx package: includes Layers,Network, Optimize
#include <oxprob.h>
#include "fashion.ox"   // the fashion data generating function

static decl Nn,K,Nh=64,Xsize=128;    // change from enum because dimensions determined  by fashion()

normalize(aB) {
    decl MaxX;
    MaxX=max(aB[0]);
    aB[0] = (2*aB[0]-MaxX) / MaxX;          //normalize to (-1,1)
    }

main() {
    decl net,batch,target,layer0,layer1,W,opt,shuffle;
    [target,batch] = get_fashion("train");   
    normalize(&batch);
    println(moments(batch,2)');
    shuffle = ranindex(rows(batch));
    Nn = columns(batch);
    batch = batch[shuffle][];                //shuffle rows
    target = target[shuffle];
    K = columns(unique(target));             // # of classes
    layer0 =       zeros(1,Nh)
            | 0.01*rann(Nn,Nh);             //stack weights under bias
    layer1 =      zeros(1,K)
            |0.01*rann(Nh,K);   
    W =loadmat("fashionW.mat");
    if (!ismatrix(W)) {
        println("fashion param file not found, resetting");
        W = vecr(layer0)|vecr(layer1);          //vectorize all parameters (will be reshaped interally)
        }
    net = new Network(CELoss);                                      // set Loss as Cross Entropy (multinomial logit));

	net.AddLayers(
        new Dense( Nn~Nh, RecLinAct),            //add the RecLinAct layer.
        new Dense( Nh~K,  SoftAct )              // default lambda = 0.0 only 2 arguments
        );
    opt = new momSGD(net,1.0,1E-3,0.9);
    opt.itmax = 1;
    net.Train(opt,&W,batch,target,Xsize);
    savemat("fashionW.mat",W);
    net.PREDICTING = TRUE;
    net.Obj(W);
    println("In sample Accuracy:",net.Loss.accuracy," Penalty: ",net.penalty);

    /*  Create an out-of-sample data set to validate */
    [target,batch] = get_fashion("test");         
    normalize(&batch);
    println(moments(batch,2)');
    net.SetBatchAndTarget(batch,target);
    net.Obj(W);
    println("Accuracy Out of Sample:",net.Loss.accuracy," Penalty: ",net.penalty);
    }
