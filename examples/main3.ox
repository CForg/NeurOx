#import "Network"
#include "spiral.ox"

// See page 382 of NNFS
main() {
	decl Xspiral,W,L,N=1000,K=3,tag;
	tag = "spiral_"+sprint(K)+"_"+sprint(N);
	Xspiral = loadmat(tag+".dta");
	if (!isint(Xspiral)) 			//data file not loaded, create data 
		Xspiral = spiral(N,K);
	net = new Network();
	net->AddLayers(new LayerDense(<2,64>,RecLinAct,5E-4),
			       new LayerDropout(0.0),	
	               new LayerDense(<64,3>,SoftAct));
	net->SetBatchAndTarget(CELoss,Xspiral[][1:],Xspiral[][0]);
	W = loadmat(tag+".mat");
//	W=0;  //reset
	if (isint(W))	//parameter file not found, start random
		W=-1+2*ranu(net.Nweights,1);
/*	net.VOLUME= TRUE;
	net->Obj(W);
//	net->Backward();
	net.VOLUME= FALSE;
//	return;
	decl G;
//	Num1Derivative(Lobj,W,&G);
//	println(G~net.grad);
//	return;	
*/
	MaxControl(300,50);
	MaxBFGS(Lobj,&W,&L,0,0);
	savemat(tag+".mat",W);
	
	net.VOLUME= TRUE;
	LayerDropout::DROPOUT = FALSE;
	net->Obj(W);
	return;
	}