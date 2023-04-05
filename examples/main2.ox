#import "NeurOx"
#include "spiral.ox"

main() {
	decl Xspiral,W,L,G;
	
	Xspiral = spiral(100,3);
	net = new Network();
	net->AddLayers(new Dense(<2,2>,RecLinAct),
	               new Dense(<2,3>,SoftAct));
	net->SetBatchAndTarget(CELoss,Xspiral[][1:],Xspiral[][0]);
	W = ranu(net.Nweights,1);
	net.VOLUME= FALSE;
/*	Lobj(W,&L,0,0);
	Num1Derivative(Lobj,W,&G);
	net->Backward();
	println(G~net.grad);
	return;
*/
//	net.BACKPROPAGATION = FALSE;
	MaxControl(300,50);
	MaxBFGS(Lobj,&W,&L,0,0);
}