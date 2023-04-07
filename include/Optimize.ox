#include "Optimize.oxh"

/** Creator for base Optimize objects .
@param net Neural Network object
**/
Optimize::Optimize(net) {
    if (!isclass(net,"Network"))
        oxrunerror("Optimize object must be a Network");
    this.net = net;
    toler = 1E-5 ;
    itmax = 1000;
    }

/**Stochastic Gradient Descent. 
@param net Neural Network object
@irate initial learning rate (default=1.0)
@decay decay rate (default-0.0)
@usemomentum default=FALSE
**/
SGD::SGD(net,irate,decay,momentum) {
    Optimize(net);
    this.irate = irate;
    this.decay = decay;
    this.momentum = momentum;
    } 

/** Iterate on SGD.
@param aW  address of starting vector of parameters. On ouput final values will replace the final values.
**/
SGD::iterate(aW) {
    decl done,step,curmom,crate,L;
    if (!isarray(aW)) oxrunerror("send a pointer to a vector: put & before initial vector");
    iter = 0;
    crate = irate;
    curmom =  zeros(aW[0]);
    do {    
        L = net.Obj(aW[0]) ;  
        net.Backward();
        step = curmom - crate * net.grad/net.BatchSize;   //divide by BatchSize to normalize
        aW[0] += step;
        done = isfeq(norm(step),0.0,toler);
        if (!imod(iter,100)) 
            println(iter," ",L," ",crate," ",norm(net.grad,2)," ",norm(curmom,2));
        if (!done) {
            crate *= 1/(1+decay);
            curmom = momentum*step;
            }
    } while (!done && ++iter < itmax);
    println("\nIteration complete");
}

AdaGrad::iterate(aW) {

}

Adam::iterate(aW) {

}


BFGS::BFGS(net) {
 Optimize(net);  
 ::net = net; 
}

/** Iterate on 
@param aW  address of starting vector of parameters
**/
BFGS::iterate(aW) {
    decl L;
    MaxBFGS(Lobj,aW,&L,0,0);
    }
