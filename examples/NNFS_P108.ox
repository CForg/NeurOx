/** Replicate the code on Page 108 of NNFS usng NeurOx 
    Output produced is given below main()
**/
#import "NeurOx"        //import the NeurOx package
#include "spiral.ox"   // the spiral generating function

enum{Nk=100,Nn=2,K=3}       // Can change the dimensions here, better than multiple "3s" in the code
main() {
    decl net,Xspiral,batch,target;
    
  	Xspiral = spiral(Nk,K);

    target = Xspiral[][0];
    batch = Xspiral[][1:2];
   
    net = new Network();                                                     //create a network
	 
    net.AddLayers(new Dense(<Nn,3>,RecLinAct, 0.0, 0, 0.01*rann(Nn,3)) );    //add the RecLinAct layer,lambda=0.0, default bias, random weights
   
    net.AddLayers(new Dense(<3,K>,SoftAct,   0.0, 0, 0.01*rann(3,K) ));     // add the Softmax Act layer

    net.SetLoss();                                                          // set Loss as "NoLoss" 
	 
    net.SetBatchAndTarget(batch,target);                                  // feed in batch and target
	 
    net->VOLUME = TRUE;
	 
    net->Forward();
    println("Output of the layer:", 
            "%12.8f",net.Loss.inputs[:5][]);                                //"%12.8f" says to print 8 decimal places

}
/** Should produce this output.  NOte the random number generator produces different weights than numpy
<pre>
Ox 9.06 (Windows_64/Parallel) (C) J.A. Doornik, 1994-2022 (oxlang.dev)
Layers 1. Total parmams: 9
Layers 2. Total parmams: 21
Output of the layer:
  0.33333333  0.33333333  0.33333333
  0.33333322  0.33333324  0.33333355
  0.33333314  0.33333317  0.33333369
  0.33333307  0.33333311  0.33333382
  0.33333295  0.33333301  0.33333403
  0.33333282  0.33333290  0.33333428

</pre>
**/