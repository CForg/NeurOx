/** Replicate code on Page 407 of NNFS usng NeurOx 
**/
#import "NeurOx"        //import the NeurOx package: includes Layers,Network, Optimize
#include "sine.ox"   // the spiral generating function

enum{Nn=1,Nh=64}             //1 input, 64 neurons 
main() {
    decl net,batch,target,layer0,layer1,layer2,W,opt;

    [target,batch] = sine();       
    println(target~batch);

    layer0 =       zeros(1,Nh)
            | 0.01*rann(Nn,Nh);             
    layer1 =      zeros(1,Nh)                            //hard-coded 1 because it is binary logit
            |0.01*rann(Nh,Nh);   
    layer2= zeros(1,1) 
            | 0.01*rann(Nh,1);
    W = vecr(layer0)|vecr(layer1)|vecr(layer2);                    

    net = new Network(MSELoss);         // set Loss as Mean Squared Error (OLS);

    net.AddLayers(
        new Dense( <Nn,Nh>, RecLinAct ),            //add the RecLinAct layer.
        new Dense(<Nh,Nh>, RecLinAct ),             //hidden linear is also Rec Lin
        new Dense( <Nh,1>,  LinAct)                // final layer is linear 
        );
    net.SetBatchAndTarget(batch,target);
    net.SetParameters(W); 
    //   opt = new BFGS(net);     MaxControl(1000,20); 
    opt = new SGD(net,1.0,1E-3,0.9);
    opt.itmax = 100;
    opt.iterate(&W); 
    net.PREDICTING = TRUE;
    net.Obj(W);
    println("In sample Accuracy:","%8.6f",net.Loss.accuracy," Penalty: ",net.penalty);

    }

/* SHould produce this output.  


*/