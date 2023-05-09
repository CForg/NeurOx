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

Optimize::initalg(inW) {
    iter = 0;
    if (!isarray(inW)) oxrunerror("send a pointer to a vector: put & before initial vector");
    this.aW = inW;
    delta = zeros(inW[0]);
    }

/** Iterate on an optimizer using virtual functions to specialize.
@param aW  address of starting vector of parameters. On ouput final values will replace the final values.
**/
Optimize::iterate(inW) {
    decl done,L;
    this.initalg(inW);
    do {    
        L = net.Obj(aW[0]) ;  
        net.Backward();
        done = this.step();            //use object's step function
        ++iter;
        if (!imod(iter,100)) 
            println(iter," ",L," ",norm(net.grad,2));
    } while (!done && iter < itmax);
}


/**Stochastic Gradient Descent. 
@param net Neural Network object
@irate initial learning rate (default=1.0)
@decay decay rate (default-0.0)
@usemomentum default=FALSE
**/
SGD::SGD(net,irate,decay) {
    Optimize(net);
    this.irate = irate;
    this.decay = decay;
    } 

SGD::initalg(inW) {
    Optimize::initalg(inW);
    crate = irate;
    }

SGD::step() {
    delta[] = - crate * net.grad/net.BatchSize;   //divide by BatchSize to normalize
    aW[0] += delta;
    crate *= 1/(1+decay);
    return FALSE;               //no criterion for stopping
    }

momSGD::momSGD(net,irate,decay,momentum) {
    SGD(net,irate,decay);
    this.momentum = momentum;
    }

momSGD::initalg(aW) {
    SGD::initalg(aW);
    curmom =  zeros(aW[0]);
    }

momSGD::step() {
    SGD::step();
    delta[] += curmom;             //adjust SGD step for momentum
    aW[0] += curmom;             // fix up change in parameters
    curmom[] = momentum*delta;   //  update momentum
    return FALSE;               //no criterion for stopping
    }

AdaGrad::AdaGrad(net,irate,decay,eps) {
    momSGD(net,irate,decay,0.0);
    this.eps = eps;
    rho = 0.0;
    }

AdaGrad::initalg(aW) {
    SGD::initalg(aW);
    cache = zeros(aW[0]);
    }

AdaGrad::step() {
    cache[] += rho*cache + (1-rho)*sqr(net.grad/net.BatchSize);
    delta[] = -crate * (net.grad/net.BatchSize) ./ (sqrt(cache)+eps);   //divide by BatchSize to normalize
    aW[0] += delta;
    crate *= 1/(1+decay);
    return FALSE;               //no criterion for stopping
    }

Adam::Adam(net,irate,decay,eps,beta1,beta2) {
    momSGD(net,irate,decay);
    this.momentum = 0.0;
    this.beta1=beta1;
    this.beta2=beta2;
}

Adam::initalg(aW) {
    momSGD::initalg(aW);        //initializes momentum and learning rate
    cache = zeros(aW[0]);        
    cbeta1 = beta1;
    cbeta2 = beta2;
    }
Adam::step() {
    curmom[] = beta1*curmom + (1-beta1)*    net.grad/net.BatchSize;
    cache[] += beta2*cache  + (1-beta2)*sqr(net.grad/net.BatchSize);
    delta[] = -crate *  (curmom / (1-cbeta1))
                     ./ (sqrt(cache)+eps);
    aW[0] += delta;
    crate *= 1/(1+decay);
    cbeta1 *= beta1;
    cbeta2 *= beta2;
    }

RMSProp::RMSProp(net,irate,decay,eps,rho) {
    AdaGrad(net,irate,decay,eps);
    this.rho = rho;
    }

BFGS::BFGS(net) {
 Optimize(net);  
 ::net = net; 
}

BFGS::initalg(aW) {
    Optimize::initalg(aW);
    }

/** Iterate on 
@param aW  address of starting vector of parameters
**/
BFGS::iterate() {
    decl L;
    MaxBFGS(Lobj,aW,&L,0,0);
    }
