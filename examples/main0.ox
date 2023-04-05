#import "NeurOx"

main() {
	net = new Network();
	net->AddLayers(new Dense(<3,2>,LinAct),
				   new Dense(<2,1>,SMAct));
	net->SetBatchAndTarget(MSELoss,ranu(4,3),matrix(ranu(4,1)));
	//net->VOLUME = TRUE;
	net->Forward();
	net->Backward();
	println("%r",net.vLabels,net.grad);
}