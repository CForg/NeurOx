/** Neural Network and Loss Functions.**/
#include "Network.oxh"


/** Create a new empty network. 
@param LossType integer code for loss type
**/
Network::Network(LossType) {
	vLabels = layers = {};
	VOLUME = isbuilt = Nlayers = Nparams = 0;
	BACKPROPAGATION = TRUE;
	this.LossType = LossType;
	switch_single(LossType) {
		case NoLoss : Loss = new Loss();
		case BinaryCELoss : Loss = new BinaryCrossEntropy();
		case CELoss : Loss = new CrossEntropy();
		case MSELoss : Loss = new MeanSquareError();
		case MAELoss : Loss = new MeanAbsoluteError();
		default : oxrunerror("Loss Type Invalid");
		}

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
					newlabs = {sprint(Nlayers)+"b"+sprint(k)}; 
				else 
					newlabs |= sprint(Nlayers)+"b"+sprint(k);
			for (n=0;n<l.Dims[Ninputs];++n)
				for(k=0;k<l.Dims[Nneurons];++k) 
					newlabs |= sprint(Nlayers)+"w"+sprint(n)+"_"+sprint(k);
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
	if (Nlayers)
		layers[.last].next = 0;  //last layer has no successor until built
	}



/** Set  batch inputs and target outputs .
@param batch TxN matrix of training data
@param target TxM matrix of target outputs

This builds the dimensions of  matrices. The code then reuses these matrices (placing [][] ) to avoid new memory 
allocation.

**/
Network::SetBatchAndTarget(batch,target)	{
	if (isbuilt)
		oxwarning("Network already built: resizing input and output");
	if (!isclass(Loss)) 
		oxrunerror("Set the Loss type using SetLoss() before setting batch and target.");
	if (LossType!=NoLoss && rows(batch)!=rows(target))
		oxrunerror("rows of batch and target must be equal");
	if (columns(batch)!=layers[0].Dims[Ninputs])
		oxrunerror("external batch input wrong dimension");
	if (LossType!=NoLoss && rows(target)!=layers[.last].Dims[Nneurons])
		oxwarning("rows of target not equal to neurons at top layer");

	/*Pre-populate inputs and outputs to avoid recreation of matrices*/
	decl l;
	layers[0].inputs = batch; 	// Make batch the inputs to the first layer
	layers[.last].next = Loss;  // Make successor of last layer the loss
	BatchSize = rows(batch);
	Loss.SetTarget(target);
	Loss.B = zeros(BatchSize,layers[.last].Dims[Nneurons]);			
	foreach (l in layers) {
		if (isclass(l,"Dropout"))
			l.curdrops = ones(l.inputs);			//l,l.Dims[Nneurons]
		l.B = l.next.inputs = zeros(rows(l.inputs),l.Dims[Nneurons]); 
		}
	grad = zeros(Nparams,1);
	isbuilt = TRUE;
	}

/** Set all biases and weights in the network. 
@param vW vectorized bias vectors and weights.

This calls SetParameters for each layer.

It also computes the regulation penalty for the network.

**/
Network::SetParameters(vW) {
	decl l;
	penalty = 0.0;        //initialize penalty
	foreach (l in layers)  {
		penalty += l->SetParameters(vW);
		if (VOLUME) println("Cumulative Penalty:",penalty);
		}
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
	lastdone = rows(grad);	//work from bottom up in the gradient matrix
	l = layers[.last];		// work backwards in the linked list of layers
	l.B[][] = Loss.B;		//B for chain rule with gradient of loss function 
	while (isclass(l)) {
		if (l.NW) { 		// layer has some parameters
			grad[lastdone-l.NW:lastdone-1] = l->Backward();  //insert vectorized gradient in right location
			lastdone -= l.NW;								 //move up the vector
			}
		else	
			l->Backward();									// no parameters layer
		l = l.prev;											//move backward in the network
		}
	}

/*------------------------------  Objective Functions  -------------------------------- */

/** Objective function of the network: Loss + penalty.
@param vW vector of biases aand weights
@return Loss + penalty

**/
Network::Obj(vW) {
	SetParameters(vW);			//populate new parameters
	Forward();				   //move forward
	return (floss + penalty);  //return overall objective
	}


/** Interface to Ox's optimization library.
See Ox's maxmize Package for explanation.

If Network::BACKPROGRATION is TRUE then the gradient is computed uisng Backward()
Otherwise Num1Derivative() is used to compute numerical gradient

@param vW vector of weights and biases
@param aL address to place -Loss 
@param aG 0 or address to place Gradient if called for
@param aH 0 or address to place Hessian if called for


@comment
This version does not compute H

@return 1

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

/*------------------------------  Loss Functions  -------------------------------- */


/** Store the target for loss functions.
@param target T x M matrix of target data
**/
Loss::SetTarget(target) {
	this.target = target;
	}

Loss::value() {
    B[][] = 1.0;
	return ;
	}

/**  
**/
RegressionData::SetTarget(target) {
	Loss::SetTarget(target);
	accuracy_precision = sqrt(varc(target)) / 250;
	}


/** Compute MSE loss and initialize B.
$$L = \sum_{t} {(y-\hat y)}^2 / 2.$$
B initialized as the gradient, (y-\hat y)
@return L
**/
MeanSquareError::value() {
	B[][] = target-inputs';
	loss = sumsqrc(B)/2.0;
	if (Network::PREDICTING) {
		//prediction =  ;
		accuracy = double(meanc(fabs(B) < accuracy_precision));
		}
	return double(loss);
	}

/** Compute MSE loss and initialize B.
$$L = \sum_{t} {(y-\hat y)}^2 / 2.$$
B initialized as the gradient, (y-\hat y)
@return L
**/
MeanAbsoluteError::value() {
	B[][] = target-inputs';
	loss = sumc(fabs(B));
	if (Network::PREDICTING) {
		//prediction =  ;
		accuracy = double(meanc(fabs(B) < accuracy_precision));
		}
	B[][] = B .> 0 .? 1 .: 0 ;  // use element-by-element conditional assignment
	return double(loss);
	}

/** Create a CrossEntropy (multinomial logit) loss function.
@param target T x M matrix of target data
**/
CrossEntropy::SetTarget(target) {
	Loss::SetTarget(target);
	J = maxc(target);								//number of classes
	rng = range(0,rows(target)-1);					// 0...T-1:  used when computing value
	targcol = reshape(range(0,J),rows(target),J+1);  // repeated rows 0 1 2 ... J-1		
	targcol = target.==targcol;						//indicator that this column is the target
	vL = zeros(rows(target),1);
	}

/** Compute MSE loss and initialize B.

@return  L  
**/
CrossEntropy::value() {
	vL[] =selectrc(inputs,rng,target)';		
	if (Network::PREDICTING) {
		prediction = maxcindex(inputs');
		accuracy = double(meanc(prediction'.==target));
		}
	loss = -sumc(log(vL));				//Don't average like NNFS, just sum
	B[][] = -(targcol-inputs);			// Jacobian of objective. vL .* and ./ vL cancel out	
	return double(loss);
	}
	
/** 
**/
BinaryCrossEntropy::SetTarget(target) {
	CrossEntropy::SetTarget(target);
	vL = zeros(target);
	}

BinaryCrossEntropy::value() {
	vL[]  = target.*inputs + (1-target).*(1-inputs);
	B[][] = -(target-inputs);
	loss = -sumc(log(vL)); 
	//println(loss,target~inputs);
	if (Network::PREDICTING) {
		prediction = inputs .> 0.5 ;
		accuracy = double(meanc(prediction.==target));
		}
	return double(loss);
	}

