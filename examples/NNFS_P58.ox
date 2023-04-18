/** Replicate  code on Page 58 of NNFS using NeurOx 
    Expected output is listed below main()
**/
#import "NeurOx"     //import the NeurOx package

main() {
    decl net,batch,biases,weights;              // Must declare variables in Ox
    batch = < 1.0, 2.0, 3.0, 2.5;               // Hard-coded matrix of inputs
              2.0, 5.0, -1.0, 2.0;
             -1.5, 2.7, 3.3, -0.8>;
    weights = <0.2, 0.8, -0.5, 1.0;
              0.5, -0.91, 0.26, -0.5;
             -0.26, -0.27, 0.17, 0.87>;
    weights = weights';                         // Neurox stores Neurons as columns in weights, so take transpose of NNFS
    biases = <2.0, 3.0, 0.5>;                   // bias ROW vector
    net = new Network();                        //create a network object. Default Loss is "NoLoss" so no target required       
	net.AddLayers(
           new Dense(<4,3>,LinAct,0.0,biases,weights)    //add Linear Act layer, lambda=0.0, populate weights and bias 
           );
	net.SetBatchAndTarget(batch);                 //  feed in batch
	net.Forward();
    println("Output of the layer:",net.Loss.inputs);    //oputs are always stored as inputs to the next level (in this case in Loss)
    }

/** Should produce this output
<pre>
Ox 9.06 (Windows_64/Parallel) (C) J.A. Doornik, 1994-2022 (oxlang.dev)
Layers 1. Total parmams 15
Output of the layer:
       4.8000       1.2100       2.3850
       8.9000      -1.8100      0.20000
       1.4100       1.0510     0.026000

</pre>
**/