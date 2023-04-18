/** Replicate the code on Page 70 of NNFS usng NeurOx 
    Expected output is listed below main()
**/
#import "NeurOx"        //import the NeurOx package
#include "spiral.ox"   // the spiral generating function

/* 
  I'm old school: avoid constants in code by giving them names. In Ox, integers can be named using enum{}
*/
enum{Nk=100,Nn=2,K=3}       // Can change the dimensions here, better than multiple "3s" in the code

main() {
    decl net,batch,target,bias,weights;     // declare all the variables used
  	[target,batch] = spiral(Nk,K);         //creates a 300 x 3 input matrix.  The [a , b ]=f()  syntax is equivalent to Python a, b = f()

    bias = zeros(1,K);
    weights = 0.01*rann(Nn,K);                    // rann() is Ox's normal pRNG

    net = new Network();                        //create a network object. Default Loss is "NoLoss" so no target required       
	  net.AddLayers(
        new Dense(<Nn,K>,LinAct,0.0,bias,weights)    // create and add Linear Act  layer, populate parameters
        );     
	  net.SetBatchAndTarget(batch,target);        
	  net->VOLUME = TRUE;                                
	  net->Forward();                                    // Forward propagation of the network
    println("Output of the layer:",
            net.Loss.inputs[:5][]);                   // output stored as input to the next level (at the top, in Loss)

}
/** Should produce this output.  NOte the random number generator produces different weights than numpy
<pre>

Ox 9.06 (Windows_64/Parallel) (C) J.A. Doornik, 1994-2022 (oxlang.dev)
Layers 1. Total parmams: 9
Output of the layer:
      0.00000      0.00000      0.00000
  -0.00010655  -1.0742e-05   5.4881e-05
  -0.00024789  -1.1193e-05   0.00016815
  -0.00037881  -1.1784e-05   0.00027256
  -0.00049727  -2.1537e-05   0.00034000
  -0.00060344  -3.5344e-05   0.00038559

</pre>
**/