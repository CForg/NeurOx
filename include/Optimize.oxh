/**Base Optimize Class **/
struct Optimize {
    const decl net;
    decl 
        aW,
        toler, 
        delta,
        itmax, 
        iter;
            Optimize(net);
    virtual iterate(aW);
    virtual step();
    virtual initalg(inaW);
    }

/** Stochastic Gradient Descent Class.**/
struct SGD : Optimize {
    decl 
        crate,
        irate,
        decay;           
            SGD(net,irate=1.0,decay=0.0);
    virtual step();
    virtual initalg(inaW);
    }

/** SGD with momentum .**/
struct momSGD  : SGD {
    decl
        curmom,
        momentum;
    virtual step();
    virtual initalg(inaW);
    momSGD(net,irate=1.0,decay=0.0,momentum=0.0);
    }

struct AdaGrad : momSGD {
    decl eps,
         cache,
         rho  ;
            AdaGrad(net,irate=1.0,decay=0.0,eps=1E-7);
    virtual step();
    virtual initalg(aW);
    }

struct RMSProp : AdaGrad {
    RMSProp(net,irate=1.0,decay=0.0,eps=1E-7,rho=0.9);
    }

struct Adam : AdaGrad {
    decl beta1, beta2, cbeta1, cbeta2;
            Adam(et,irate=1.0,decay=0.0,eps=1E-7,beta1=.9,beta2=.999);
    virtual step();
    virtual initalg(aW);
    }

/** Inteface for Ox's MaxBFGS. 
**/
struct BFGS : Optimize {
   BFGS(net) ;
   iterate();
   initalg(aW);
   }

