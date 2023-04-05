/* This replicates the code on Page 125 of NNFS usng NeurOx */
#import "NeurOx"        //import the NeurOx package
#include "spiral.ox"   // the spiral generating function

main() {
    decl net,Xspiral,batch,target,layer0,layer1;
  	Xspiral = spiral(100,3);

    batch = Xspiral[][1:2];
    target = Xspiral[][0];

    net = new Network();                        //create a network
	net.AddLayers(new Dense(<2,3>,RecLinAct));     //add the RecLinAct layer
    net.AddLayers(new Dense(<3,3>,SoftAct));
	net.SetBatchAndTarget(CELoss,batch,target);        // set Loss as "NoLoss" so no target required, feed in batch

    layer0 =  vecr( zeros(1,3) | 0.01*rann(2,3) );
    layer1 =  vecr( zeros(1,3) | 0.01*rann(3,3) );

    net.SetParameters( layer0 | layer1 );  // Populate weights and biases (always vectorized then reshaped internally)

	net->VOLUME = TRUE;
	net->Forward();
    println("Output of the layer:","%12.8f",net.Loss.inputs[:5][]);    //outputs are always stored as inputs to the next level (in this case in Loss)
    println("Loss: ",net.floss);
    net->Backward();
    decl NumGrad;
    ::net = net;            // have to copy the network to global so it can be used by Lobj
    Num1Derivative(Lobj,layer0|layer1,&NumGrad);
    println("Full Gradient:","%c",{"BackProg","Numerical (-Loss)"},net.grad~NumGrad);
    

}
/* SHould produce this output.  NOte the random number generator produces different weights

Ox 9.06 (Windows_64/Parallel) (C) J.A. Doornik, 1994-2022 (oxlang.dev)
Layers 1. Total parmams: 9
Layers 2. Total parmams: 21
Warning in SetBatchAndTarget: rows of target not equal to neurons at top layer
Prediction Rate
      0.28333
Output of the layer:
  0.33333333  0.33333333  0.33333333
  0.33333322  0.33333324  0.33333355
  0.33333314  0.33333317  0.33333369
  0.33333307  0.33333311  0.33333382
  0.33333295  0.33333301  0.33333403
  0.33333282  0.33333290  0.33333428

Loss:
       1.0986


*/