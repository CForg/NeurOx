#include "Optimize.oxh"

Optimize::Optimize(net) {
    this.net = net;
    toler = 1E-5 ;
    itmax = 1000;
    }

SGD::SGD(net,irate,decay,usemomentum) {
    Optimize(net);
    this.irate = irate;
    this.decay = decay;
    this.usemomentum = usemomentum;
    } 

SGD::iterate(aW) {
    decl done,step,curmom,crate,L;
    iter = 0;
    crate = irate;
    curmom =  (usemomentum) ? zeros(aW[0]) : 0.0;
    do {
        L = net.Obj(aW[0]) ;  
        step = curmom - crate * net.grad;
        aW[0] += step;
        done = isfeq(norm(step),0.0,toler);
        if (!done) {
            crate *= 1/(1+decay);
            if (usemomentum) 
                curmom += step;
            }
    } while (done || ++iter==itmax);
}

AdaGrad::iterate(aW) {

}

Adam::iterate(aW) {

}

BFGS::iterate(aW) {
    decl L;
    MaxBFGS(Lobj,aW,&L,0,0);
    }
