/**Base Optimize Class **/
struct Optimize {
    const decl net;
    decl 
        toler, 
        itmax, 
        iter;
    Optimize(net);
    }

/** Stochastic Gradient Descent Class.**/
struct SGD : Optimize {
    decl 
        momentum,
        irate,
        decay;
    SGD(net,irate=1.0,decay=0.0,momentum=0.0);
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

/** Inteface for Ox's MaxBFGS. 
**/
struct BFGS : Optimize {
   BFGS(net) ;
   iterate(aW);
   }

