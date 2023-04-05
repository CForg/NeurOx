struct Optimize {
    const decl net;
    decl 
        toler, 
        itmax, 
        iter;
    Optimize(net);
    }

struct SGD : Optimize {
    decl 
        usemomentum,
        irate,
        decay;
    SGD(net,irate=1.0,decay=0.0,usemomentum=FALSE);
    iterate(aW);
    }

struct AdaGrad : Optimize {
    AdaGrad(net);
    iterate(aW);
    }

struct Adam : Optimize {
    Adam(net);
    iterate(aW);
    }

struct BFGS : Optimize {
   BFGS() ;
   iterate(aW);
   }

