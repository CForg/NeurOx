/* This replicates the code on Page 136 of NNFS usng NeurOx */
#import "NeurOx"        //import the NeurOx package
#include "spiral.ox"   // the spiral generating function

main() {
    decl net,Xspiral,batch,target,layer0,layer1,W,bestW, bestL, trip;
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
	net->Forward();
    bestL = net.floss;
    bestW = W;
    println("initial loss:",bestL);
    for(trip=0; trip<10000; ++trip) {
        W += 0.05 * rann(rows(W),1);
        net.SetParameters(W);    
	    net->Forward();
        if (net.floss<bestL) {
            bestL = net.floss;
            bestW = W;
            println("* ",trip,":",bestL);
            }
        else   
            if (!imod(trip,10)) print(".");     //progress dots every 10 trips
        }

    }
/* SHould produce this output.  NOte the random number generator produces different weights

Ox 9.06 (Windows_64/Parallel) (C) J.A. Doornik, 1994-2022 (oxlang.dev)
Layers 1. Total parmams: 9
Layers 2. Total parmams: 21
Warning in SetBatchAndTarget: rows of target not equal to neurons at top layer
initial loss:329.586
.* 5:329.223
...* 33:329.084
* 38:327.824
* 39:327.543
.....* 83:326.008
* 84:325.645
* 86:323.48
...............................................................................................................................
...............................................................................................................................
...............................................................................................................................
...............................................................................................................................
...............................................................................................................................
...............................................................................................................................
...............................................................................................................................
...................................................................................................... *  
*/