#include "Network.oxh"

Lobj(vW,aL,aG,aH) {
	aL[0] = -net->Obj(vW);	   //convert to maximum
	if (!isint(aG))	{
		if (net.BACKPROPAGATION) {
			net->Backward();
			aG[0] = -net.grad;	//Have to negate analytic loss derivative
			}
		else {
			decl tmpDO = LayerDropout::DROPOUT;
			LayerDropout::DROPOUT=FALSE;
			Num1Derivative(Lobj,vW,aG);
			LayerDropout::DROPOUT=tmpDO;
			}
		}
	return 1;
	}

Network::Obj(vW) {
	decl l;
	penalty = 0.0;
	foreach (l in layers)  {
		penalty += l->UpdateWeights(vW);
		}
	Forward();
	return (floss + penalty);  
	}
Loss::Loss(target) {
	this.target = target;
	}
MeanSquareError::MeanSquareError(target) {
	Loss(target);
	}
MeanSquareError::value() {
	B[][] = target-inputs';
	loss = sumsqrc(B)/2.0;
	return loss;
	}
CrossEntropy::CrossEntropy(target) {
	Loss(target);
	rng = range(0,rows(target)-1);
	targcol = reshape(range(0,maxc(target)),rows(target),maxc(target)+1);
	targcol = target.==targcol;
	range(0,maxc(target)-1);
	}
CrossEntropy::value() {
	vL =selectrc(inputs,rng,target)';
	B[][] = vL.*(targcol - inputs);
	if (Network::VOLUME) {
		decl pred = maxcindex(inputs') ';
		println("Prediction Rate",meanc(pred.==target));
		}
	loss = -sumc(log(vL));
	B[][] .*= -1.0 ./ vL;
	return loss;
	}
	
BinaryCrossEntropy::BinaryCrossEntropy(target) {
	CrossEntropy(target);
	}
BinaryCrossEntropy::value() {
	B[][] = (2*target-1).*inputs+(1-target);
	loss = -sumc(log(B)); //transpose once
	B[][] = -(2.0*target-1)./ B;
	return loss;
	}

/** Create a new empty network. **/
Network::Network() {
	layers = {};
	VOLUME = isbuilt = Nlayers = Nweights = 0;
	BACKPROPAGATION = TRUE;
	}

/** Add layers to the network.
@param ... each argument is a layer object.
**/
Network::AddLayers(...args) {
	decl l;
	foreach (l in args) {
		if (!isclass(l,"Layer"))
			oxrunerror("layer has to be a Layer object");
		if (isclass(l,"LayerDense")) {
			l.MyW0 = Nweights;
			Nweights += l.NW;
			}
		else if (isclass(l,"LayerDropout")) {
			if (!Nlayers)
				oxrunerror("First layer should not be Dropout");
			l.Dims = constant(layers[.last].Dims[Nneurons],1,2);  //pass thru layer
			println("DROPOUT dims",l.Dims);
			}
		if (Nlayers) {
			if (isclass(l,"LayerDense")
				&&  l.Dims[Ninputs]!=layers[.last].Dims[Nneurons])
				oxrunerror("layer "+sprint(Nlayers)+" should take "+sprint(layers[Nlayers-1].Dims[Nneurons])+" inputs");
			l.prev = layers[.last];   // double link layers
			layers[.last].next = l;	  // point to the new layer
			}
		else
			l.prev = 0; // first layer has no predecessor
		layers |= l;
		++Nlayers;		// one more layer
		println("Layers ",Nlayers,". Total parmams",Nweights);
		}
	if (Nlayers)
		layers[.last].next = 0;  //last layer has no successor until built
	}

/** Set the batch inputs and target outputs .**/
Network::SetBatchAndTarget(LossType,batch,target)	{
	if (isbuilt)
		oxrunerror("Network already built");
	isbuilt = TRUE;
	if (rows(batch)!=rows(target))
		oxrunerror("rows of batch and target must be equal");
	if (columns(batch)!=layers[0].Dims[Ninputs])
		oxrunerror("external batch input wrong dimension");
	if (rows(target)!=layers[.last].Dims[Nneurons])
		oxwarning("rows of target not equal to neurons at top layer");
	switch_single(LossType) {
		case CELoss : Loss = new CrossEntropy(target);
		case MSELoss : Loss = new MeanSquareError(target);
		default : oxrunerror("Loss Type Invalid");
		}
	/*Pre-populate inputs and outputs to avoid recreation of matrices*/
	decl l;
	layers[0].inputs = batch; 	// Make batch the inputs to the first layer
	layers[.last].next = Loss;  // Make successor of last layer the loss
	BatchSize = rows(target);
	Loss.B = zeros(BatchSize,layers[.last].Dims[Nneurons]);
	foreach (l in layers) {
		if (isclass(l,"LayerDropout"))
			l.curdrops = ones(l.inputs);			//l,l.Dims[Nneurons]
		l.output = zeros(rows(l.inputs),l.Dims[Nneurons]);
		l.next.inputs = zeros(l.output);
		l.B = zeros(l.output);		
		}
	grad = zeros(Nweights,1);
	}
	
Network::Forward() {
	decl l;
	foreach ( l in layers ) {
		l -> Forward();
		if (VOLUME>1) println("-------activation------",l.next.inputs);
		}
	floss = Loss->value();
	}
Network::Output() {
	decl l;
	foreach ( l in layers ) 
		println(l.weights);
	}
	
/** Not working yet.**/
Network::Backward() {
	decl l,lastdone = rows(grad);
	l = layers[.last];
	//gradient of the loss function initializes B for chain rule
	l.B[][] = Loss.B;
	while (isclass(l)) {
//		println("**",l.B[:20][:min(10,.last)]);
		if (l.NW) {
			grad[lastdone-l.NW:lastdone-1] = l->Backward();
			lastdone -= l.NW;
			}
		else	//Dropout layer
			l->Backward();
//		println(l.B[:20][:min(10,.last)]);			
		l = l.prev;
		}
	}