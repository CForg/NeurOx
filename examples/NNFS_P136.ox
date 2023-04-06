/* This replicates the code on Page 243 of NNFS usng NeurOx */
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
Prediction Rate
      0.25333
Output of the layer:
  0.33333333  0.33333333  0.33333333
  0.33333350  0.33333332  0.33333318
  0.33333385  0.33333330  0.33333285
  0.33333416  0.33333328  0.33333256
  0.33333437  0.33333326  0.33333236
  0.33333451  0.33333326  0.33333224

Loss:
       329.59
Full Gradient:
     BackProgNumerical (-Loss)
     0.074145    -0.074230
    -0.089461     0.088704
      0.24088     -0.24088
     0.022891    -0.022891
     0.064770    -0.064770
     -0.12189      0.12189
    -0.076757     0.076757
     -0.19451      0.19656
     0.085930    -0.085930
  -8.4037e-05   8.4060e-05
    0.0010664   -0.0010664
  -0.00098241   0.00098246
     -0.15799      0.15799
      0.12889     -0.12889
     0.029099    -0.029099
    0.0040722   -0.0040722
    0.0035600   -0.0035600
   -0.0076322    0.0076322
      0.11431     -0.11431
    -0.059412     0.059412
    -0.054902     0.054902
 


*/