/* Replicate  code on Page 58 of NNFS usng NeurOx */
#import "NeurOx"     //import the NeurOx package

main() {
    decl net,batch,biases,weights;
    batch = < 1.0, 2.0, 3.0, 2.5;
              2.0, 5.0, -1.0, 2.0;
             -1.5, 2.7, 3.3, -0.8>;
    
    weights = <0.2, 0.8, -0.5, 1.0;
              0.5, -0.91, 0.26, -0.5;
             -0.26, -0.27, 0.17, 0.87>;
    weights = weights';                 // Neurox uses Neurons as columns in weights, so take transpose
    biases = <2.0, 3.0, 0.5>;           // bias row vector
    net = new Network();                        //create a network
	net.AddLayers(
           new Dense(<4,3>,LinAct,0.0,biases,weights)    //add LinAct layer, lambda=0.0,populate weights and biases 
           );
	net.SetBatchAndTarget(NoLoss,batch);       // set Loss as "NoLoss" so no target required, feed in batch
	net.Forward();
    println("Output of the layer:",net.Loss.inputs);    //oputs are always stored as inputs to the next level (in this case in Loss)

}
/* SHould produce this output
Ox 9.06 (Windows_64/Parallel) (C) J.A. Doornik, 1994-2022 (oxlang.dev)
Layers 1. Total parmams 15
Output of the layer:
       4.8000       1.2100       2.3850
       8.9000      -1.8100      0.20000
       1.4100       1.0510     0.026000

*/