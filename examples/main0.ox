#import "NeurOx"

main() {
	net = new Network();
	net->AddLayers(new LayerDense(<3,2>,LinAct),
					new LayerDense(<2,1>,SMAct));
	net->SetBatchAndTarget(MSELoss,ranu(4,3),matrix(ranu(4,1)));
	//net->VOLUME = TRUE;
	net->Forward();
	net->Backward();
	println(net.grad);
}