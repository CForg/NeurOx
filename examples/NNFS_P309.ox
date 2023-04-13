/* 
        Replicates the code on Page 309 of NNFS usng NeurOx 
*/
#import "NeurOx"        //import the NeurOx package: includes Layers,Network, Optimize
#include "spiral.ox"   // the spiral generating function

enum{Nk=100,Nn=2,K=3,Nh=64}   // Hidden layer now has 64 neurons instead of 3

main() {
        decl net,Xspiral,batch,target,layer0,layer1,W,opt;
  	
    Xspiral = spiral(Nk,K);

    target = Xspiral[][0];
    batch = Xspiral[][1:];

    layer0 =       zeros(1,Nh)
            | 0.01*rann(Nn,Nh);     //stack weights under bias
    layer1 =      zeros(1,K)
            |0.01*rann(Nh,K);   
    W = vecr(layer0)|vecr(layer1);        //vectorize all parameters (will be reshaped interally)

    net = new Network();                        //create a network
    net.AddLayers(
        new Dense(<Nn,Nh>,RecLinAct),      //add the RecLinAct layer
        new Dense(<Nh,K>,SoftAct)
        );
    net.SetLoss(CELoss);    
    net.SetBatchAndTarget(batch,target);        
    net.SetParameters(W); 
  //   opt = new BFGS(net);     MaxControl(1000,20); 
    opt = new SGD(net,1.0,1E-3,0.9);
    opt.itmax = 10000;
    opt.iterate(&W);
    net.PREDICTING = TRUE;
    net.Obj(W);
    println("In sample Accuracy:",net.Loss.accuracy," Penalty: ",net.penalty);

    /*  Create an out-of-sample data set to validate */
    Xspiral = spiral(Nk,K);
    target = Xspiral[][0];
    batch = Xspiral[][1:];
    net.SetBatchAndTarget(batch,target);
    net.PREDICTING = TRUE;
    net.Obj(W);
    println("Accuracy Out of Sample:",net.Loss.accuracy," Penalty: ",net.penalty);

    }

/* SHould produce this output.  



*/