/* This replicates the code on Page 243 of NNFS usng NeurOx */
#import "NeurOx"        //import the NeurOx package
#include "spiral.ox"   // the spiral generating function

main() {
    decl net,Xspiral,batch,target,layer0,layer1,W;
  	Xspiral = spiral(100,3);

    batch = Xspiral[][1:2];
    target = Xspiral[][0];

    layer0 = zeros(1,3)|0.01*rann(2,3);     //stack weights under bias
    layer1 = zeros(1,3)|0.01*rann(3,3);
    W = vecr(layer0)|vecr(layer1);        //vectorize all parameters (will be reshaped interally)

    net = new Network();                        //create a network
	net.AddLayers(
        new Dense(<2,3>,RecLinAct),      //add the RecLinAct layer
        new Dense(<3,3>,SoftAct)
        );
	net.SetBatchAndTarget(CELoss,batch,target);        // set Loss as "NoLoss" so no target required, feed in batch
    net.SetParameters(W);    
	net->VOLUME = TRUE;
	net->Forward();
    println("Output of the layer:","%12.8f",net.Loss.inputs[:5][]);    //outputs are always stored as inputs to the next level (in this case in Loss)
    println("Loss: ",net.floss);
    net->Backward();
    net->VOLUME = FALSE;
    decl NumGrad;
    ::net = net;            // have to copy the network to global so it can be used by Lobj
    Num1Derivative(Lobj,W,&NumGrad);
    println("Full Gradient:","%c",{"BackProg","Numerical (-Loss)"},net.grad~NumGrad);
    }
/* SHould produce this output.  NOte the random number generator produces different weights



*/