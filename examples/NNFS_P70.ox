/* This replicates the code on Page 70 of NNFS usng NeurOx */
#import "NeurOx"        //import the NeurOx package
#include "spiral.ox"   // the spiral generating function

main() {
    decl net,Xspiral,batch,target,bias,weights;
  	Xspiral = spiral(100,3);

    batch = Xspiral[][1:2];
    target = Xspiral[][0];
    bias = zeros(1,3);
    weights = 0.01*rann(2,3);

    net = new Network();                        //create a network
	net.AddLayers(
            new Dense(<2,3>,LinAct,0.0,bias,weights)    //create layer, populate parameters
            );     //add the LinAct layer

	net.SetBatchAndTarget(NoLoss,batch,target);        // set Loss as "NoLoss" so no target required, feed in batch

	net->VOLUME = TRUE;
	net->Forward();
    println("Output of the layer:",net.Loss.inputs[:5][]);    //outputs are always stored as inputs to the next level (in this case in Loss)

}
/* SHould produce this output.  NOte the random number generator produces different weights

Layers 1. Total parmams: 9
Output of the layer:
      0.00000      0.00000      0.00000
   4.7741e-05   3.6363e-05  -7.5941e-05
   0.00023025   5.0237e-05  -0.00011809
   0.00039780   6.3522e-05  -0.00015787
   0.00046979   9.8497e-05  -0.00023300
   0.00049057   0.00014246  -0.00032185

*/