/** Neural Network.**/
#include "Network.oxh"

/** Interface to Ox's optimization library.
@param vW vector of weights and biases
@param aL address to place -Loss 
@param aG 0 or address to place Gradient if called for
@param aH 0 or address to place Hessian if called for

@return 1

Note:  This version does not compute H
If Network::BACKPROGRATION is TRUE then the gradient is computed uisng Backward()
Otherwise Num1Derivative() is used to compute numerical gradient

**/
Lobj(vW,aL,aG,aH) {
	aL[0] = -net->Obj(vW);	   //convert to maximum
	if (!isint(aG))	{
		if (net.BACKPROPAGATION) {
			net->Backward();
			aG[0] = -net.grad;	//Have to negate analytic loss derivative
			}
		else {
			decl tmpDO = Layer::DROPOUT;
			Layer::DROPOUT=FALSE;
			Num1Derivative(Lobj,vW,aG);
			Layer::DROPOUT=tmpDO;
			}
		}
	return 1;
	}

/** Set all biases and weights in the network. 
@param vW vectorized bias vectors and weights.

This calls SetParameters for each layer.

It also computes the regulation penalty for the network.


**/
Network::SetParameters(vW) {
	decl l;
	penalty = 0.0;        //initialize penalty
	foreach (l in layers)  
		penalty += l->SetParameters(vW);
}

/** Objective  function of the network
@param vW vector of biases aand weights
@return Loss + penalty

**/
Network::Obj(vW) {
	SetParameters(vW);
	Forward();
	return (floss + penalty);  
	}

/** Base Loss function creator.
@param target T x M matrix of target data
**/
Loss::Loss(target) {
	this.target = target;
	}

Loss::value() {
    B[][] = 1.0;
	return ;
	}

/** Create a mean squared error loss function.
@param target T x M matrix of target data
**/
MeanSquareError::MeanSquareError(target) {
	Loss(target);
	}

/** Compute MSE loss and initialize B.
$$L = \sum_{t} {(y-\hat y)}^2 / 2.$$
B initialized as the gradient, (y-\hat y)
@return L
**/
MeanSquareError::value() {
	B[][] = target-inputs';
	loss = sumsqrc(B)/2.0;
	return loss;
	}

/** Create a CrossEntropy (multinomial logit) loss function".
@param target T x M matrix of target data
**/
CrossEntropy::CrossEntropy(target) {
	Loss(target);
	rng = range(0,rows(target)-1);
	targcol = reshape(range(0,maxc(target)),rows(target),maxc(target)+1);
	targcol = target.==targcol;
	range(0,maxc(target)-1);
	}

/** Compute MSE loss and initialize B.

@return  L  
**/
CrossEntropy::value() {
	vL =selectrc(inputs,rng,target)';
	B[][] = vL.*(targcol - inputs);
	if (Network::VOLUME) {
		decl pred = maxcindex(inputs');
		println("Prediction Rate",meanc(pred'.==target));
		}
	loss = -sumc(log(vL));
	B[][] .*= -1.0 ./ vL;
	return loss;
	}
	
/** 
**/
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
	vLabels = layers = {};
	VOLUME = isbuilt = Nlayers = Nparams = 0;
	BACKPROPAGATION = TRUE;
	}

/** Add layers to the network.
@param ... each argument is a layer object.
**/
Network::AddLayers(...args) {
	decl l,n,k,newlabs;
	foreach (l in args) {
		if (!isclass(l,"Layer"))
			oxrunerror("layer has to be a Layer object");
		if (isclass(l,"Dense")) {
			l.MyW0 = Nparams;
			Nparams += l.NW;
			for(k=0;k<l.Dims[Nneurons];++k) 
				if (!k) 
					newlabs = {"b"+sprint(Nlayers)+"_n"+sprint(k)}; 
				else 
					newlabs |= "b"+sprint(Nlayers)+"_n"+sprint(k);
			for (n=0;n<l.Dims[Ninputs];++n)
				for(k=0;k<l.Dims[Nneurons];++k) 
					newlabs |= "w"+sprint(Nlayers)+"_i"+sprint(n)+"_n"+sprint(k);
			vLabels |= newlabs;
			}
		else if (isclass(l,"Dropout")) {
			if (!Nlayers)
				oxrunerror("First layer should not be Dropout");
			l.Dims = constant(layers[.last].Dims[Nneurons],1,2);  //pass thru layer
			println("DROPOUT dims",l.Dims);
			}
		if (Nlayers) {
			if (isclass(l,"Dense")
				&&  l.Dims[Ninputs]!=layers[.last].Dims[Nneurons])
				oxrunerror("layer "+sprint(Nlayers)+" should take "+sprint(layers[Nlayers-1].Dims[Nneurons])+" inputs");
			l.prev = layers[.last];   // double link layers
			layers[.last].next = l;	  // point to the new layer
			}
		else
			l.prev = 0; // first layer has no predecessor
		layers |= l;
		++Nlayers;		// one more layer
		println("Layers ",Nlayers,". Total parmams: ",Nparams);
		}
	println("Labels",vLabels);
	if (Nlayers)
		layers[.last].next = 0;  //last layer has no successor until built
	}

/** Set the Loss type, batch inputs and target outputs .
@param LossType integer code for Loss
@param batch TxN matrix of training data
@param target TxM matrix of target outputs

This builds the network and initializes the dimensions of  matrices. 
The code then reuses these matrices (placing [][] ) to avoid new memory 
allocation.

**/
Network::SetBatchAndTarget(LossType,batch,target)	{
	if (isbuilt)
		oxrunerror("Network already built");
	isbuilt = TRUE;
	if (LossType!=NoLoss && rows(batch)!=rows(target))
		oxrunerror("rows of batch and target must be equal");
	if (columns(batch)!=layers[0].Dims[Ninputs])
		oxrunerror("external batch input wrong dimension");
	if (LossType!=NoLoss && rows(target)!=layers[.last].Dims[Nneurons])
		oxwarning("rows of target not equal to neurons at top layer");
	switch_single(LossType) {
		case NoLoss : Loss = new Loss(target);
		case CELoss : Loss = new CrossEntropy(target);
		case MSELoss : Loss = new MeanSquareError(target);
		default : oxrunerror("Loss Type Invalid");
		}
	/*Pre-populate inputs and outputs to avoid recreation of matrices*/
	decl l;
	layers[0].inputs = batch; 	// Make batch the inputs to the first layer
	layers[.last].next = Loss;  // Make successor of last layer the loss
	BatchSize = rows(batch);
	Loss.B = zeros(BatchSize,layers[.last].Dims[Nneurons]);
	foreach (l in layers) {
		if (isclass(l,"Dropout"))
			l.curdrops = ones(l.inputs);			//l,l.Dims[Nneurons]
		l.B = l.next.inputs = zeros(rows(l.inputs),l.Dims[Nneurons]); 
		}
	grad = zeros(Nparams,1);
	}

/**Forward propagate the network.**/	
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
	
/** Back Propgation .**/
Network::Backward() {
	decl l,lastdone;
	lastdone = rows(grad);	//work backwards in the W matrix
	l = layers[.last];
	//gradient of the loss function initializes B for chain rule
	l.B[][] = Loss.B;
	while (isclass(l)) {
		if (l.NW) { // insert new layer gradients in the overall vector
			grad[lastdone-l.NW:lastdone-1] = l->Backward();
			lastdone -= l.NW;
			}
		else	//Dropout layer
			l->Backward();
		l = l.prev;
		}
	}
