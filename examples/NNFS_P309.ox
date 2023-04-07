/* This replicates the code on Page 309 of NNFS usng NeurOx */
#import "NeurOx"        //import the NeurOx package: includes Layers,Network, Optimize
#include "spiral.ox"   // the spiral generating function

main() {
    decl net,Xspiral,batch,target,layer0,layer1,W,opt,dims;
  	Xspiral = spiral(100,3);

    batch = Xspiral[][1:2];
    target = Xspiral[][0];

    dims = <2,64; 64, 3>;                   // store dimensions of layers
    layer0 =       zeros(1,dims[0][Nneurons])
            | 0.01*rann(dims[0][Ninputs],dims[0][Nneurons]);     //stack weights under bias
    layer1 =      zeros(1,dims[1][Nneurons])
            |0.01*rann(dims[1][Ninputs],dims[1][Nneurons]);   
    W = vecr(layer0)|vecr(layer1);        //vectorize all parameters (will be reshaped interally)

    net = new Network();                        //create a network
	net.AddLayers(
        new Dense(dims[0][],RecLinAct),      //add the RecLinAct layer
        new Dense(dims[1][],SoftAct)
        );
	net.SetBatchAndTarget(CELoss,batch,target);        // set Loss as "NoLoss" so no target required, feed in batch
    net.SetParameters(W); 
  //   opt = new BFGS(net);     MaxControl(1000,20); 
    opt = new SGD(net,1.0,1E-3,0.9);
    opt.itmax = 10000;
    opt.iterate(&W);
    net.PREDICTING = TRUE;
    net.Obj(W);
    println("Accuracy:",net.Loss.accuracy);
    }

/* SHould produce this output.  NOte the random number generator produces different weights



*/