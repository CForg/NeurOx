/**
Replicate the code on Page 136 of NNFS usng NeurOx 
**/
#import "NeurOx"        //import the NeurOx package
#include "spiral.ox"   // the spiral generating function

enum{Nk=100,Nn=2,K=3,Nh=3}   

main() {
    decl net,Xspiral,batch,target,layer0,layer1,W,bestW, bestL, trip;
  	Xspiral = spiral(Nk,K);

    target = Xspiral[][0];
    batch = Xspiral[][1:];


    layer0 = zeros(1,Nh)|0.01*rann(Nn,Nh);     //stack weights under bias
    layer1 = zeros(1,K)|0.01*rann(Nh,K);
    W = vecr(layer0)|vecr(layer1);        //vectorize all parameters (will be reshaped interally)

    net = new Network();                        //create a network
	net.AddLayers(
        new Dense(<Nn,Nh>,RecLinAct),      //add the RecLinAct layer
        new Dense(<Nh,K>,SoftAct)
        );
    net.SetLoss(CELoss);                                // set Loss as Cross Entropy (multinomial logit)
	net.SetBatchAndTarget(batch,target);        
    net.SetParameters(W);    
	net->Forward();
    bestL = net.floss;                                // initialize best Loss so far
    bestW = W;                                        // and best W vector so far
    println("initial loss:",bestL);
    for(trip=0; trip<10000; ++trip) {
        W += 0.05 * rann(rows(W),1);                 // random search (so dumb...should use Amoeba!)
        net.SetParameters(W);    
	    net->Forward();
        if (net.floss<bestL) {                      // is the current value better than the best so far?
            bestL = net.floss;                      
            bestW = W;
            println("* ",trip,":",bestL);
            }
        else   
            if (!imod(trip,10)) print(".");       //progress dots every 10 trips
        }

    }

/* SHould produce this output.  NOte the random number generator produces different weights
<pre>
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

</pre>
*/