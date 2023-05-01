/** Layers of Neurons and Activations. **/
#include "Layers.oxh"

/**Return dimensions of a matrix as a 1x2 vector. **/
Layer::Dimensions(A)   { return rows(A)~columns(A); }


/** Create a dense layer in a neural network.
@param dims =  <Ninputs,Nneurons> dimensions of the layer
@param ActType integer code for the Activation
@lambda absolute value regularization factor [default=0.0]
@ibias initial bias [default 0 = vector of zeros]
@iweights initial bias [default 0 = equal normalized weights]

**/
Dense::Dense(Dims,ActType,lambda,ibias,iweights) {
	this.Dims = Dims;
	this.lambda = lambda;
	bias = isint(ibias) 
				? zeros(1,Dims[Nneurons]) 
				: ibias;
	weights = isint(iweights) 
				? constant(1/Dims[Ninputs],Dims[Ninputs],Dims[Nneurons]) 
				: iweights;
	if (Dimensions(bias)!=1~Dims[Nneurons]) 
		oxrunerror("initial bias vector not right dimensions");
	if (Dimensions(weights)!=Dims) 
		oxrunerror("initial weight matrix not right dimensions");		
	NW = int(Dims[Nneurons]+prodr(Dims));  //# of weights & bias parameters
	GP = zeros(NW,1);		 			  //scratch space for vectorized parameters AND gradient of penalty
	GM = zeros(weights)|0;  			  // matrix for bias and weight loss gradients
	switch_single (ActType) {
		case LinAct 		:  Activation = Linear;
		case RecLinAct		:  Activation = RectLinear;
		case SMAct			:  Activation = Sigmoid;
		case SoftAct		:  Activation = SoftMax;
		default				:  oxrunerror("Activation Type Invalid, see integer codes in enum{}");
		}
    }

/** Update Layer's weights and bias from the overall parameter vector.
@param newVW  vector of all weights and biases in the network

This copies elements of newVW from MyW0 to MyW0+NW-1 and reshapes them
into bias vector and weight matrix.

@comment
	Currently the same &lambda; for bias and weights at a given layer

@return the "regularization" penalty for the weights
**/
Dense::SetParameters(newVW) {
	GP[] = newVW[MyW0:MyW0+NW-1];			//copy into scratch space my parameters
	bias[] = GP[:Dims[Nneurons]-1];
	weights[][] = reshape(GP[Dims[Nneurons]:.last], Dims[Ninputs], Dims[Nneurons] );
	decl penalty;
	if (lambda>0.0) { //only compute penalty  if necessary
		penalty = lambda*(		
				         norm(GP,1)					// sum of |x|
					+sqr(norm(GP,2))        		// NNFS does not take square root, so square Euclidean norm
					);
		GP[] = lambda*(
				(GP .> 0  .? 1.0 .:  -1.0)		// derivative of the absolute function penalty.  KINK at 0.0!
				+ 2*GP						// derivative of sum of squares
	    		);
		}
	else {										// no penalty or gradient
		 penalty = 0.0;
		 GP[] = 0.0;
		}
	return penalty;
	}
	
/** Forward propogation.
Compute $b+xA$, send to the Activation, initialize B.
**/
Dense::Forward() {
	Activation(bias + inputs*weights);
	}

/** Back Propagation. 
Compute gradients of weights then update B in the previous level
to be used as forward part of gradient 

This also adds the gradient of the penalty (if any)
**/
Dense::Backward() {
	decl t;

	// aggregate across observations in the batch.  
	for (GM[][] = 0.0, t=0 ; t<rows(inputs); ++t) 
		GM += (1~inputs[t][])'*B[t][];			//1 is coeff on bias, 
	if (isclass(prev)) // not the first layer so
		prev.B[][] .*= B*weights';		//continue chain rule, multiply by weights' & previous layer's activation
	return vecr(GM)+GP;   			   // vectorize Jacobian matrix and add on the gradient from the penalty
	}

Dense::Plot() {
	}

/** Create a dropout layer.
@param rate fraction of neurons to leave on (turnoff 1-rate)
**/
Dropout::Dropout(rate){
	this.rate = rate;
	NW = 0;  		 // no weights in  a dropout layer
	DROPOUT = TRUE;  // turn on drop out if a layer exists
	}

/**Do nothing and return 0 penalty.
**/
Dropout::SetParameters(newvW) {
	return 0.0;   
	}

/**Zero out a random set of neuron activations.
**/
Dropout::Forward() {
	curdrops[][] = (DROPOUT) ?
							ranu(rows(curdrops),columns(curdrops)).>rate  //1,columns(curdrops)
							: 1.0;	
	next.inputs[][] = curdrops .* inputs / (1-DROPOUT*rate);
	B[][] = (next.inputs.>0)/(1-DROPOUT*rate);
	}

/**Update B **/
Dropout::Backward() {
	prev.B[][] .*= B;	// multiply my activation gradient x future effects
	}

/*------------------------------  Activations -------------------------------- */


/**Linear Activation.
@param output.  TxN matrix of neuron outputs
Activations store values as input in the next layer. 
$$z = y = b + xA$$
B is initialize as one.
**/
Layer::Linear(output)		{
	next.inputs[][] = output;	//just pass output 
	B[][] = 1.0;
	}

/**Linear Activation.
@param output.  TxN matrix of neuron outputs
z is stored as input in the next layer.
$$z = {1 \over 1+e^{y}}$$
$$B = z *(1-z)$$
**/
Layer::Sigmoid(output) 	{
	next.inputs[][] =  1.0 ./(1+exp(-output));
	B[][] = next.inputs.*(1-next.inputs);		//dF = F(1-F)
	}

/**Rectilnear Activation.
@param output.  TxN matrix of neuron outputs
z is stored as input in the next layer.
$$z = \max{0,y}$$
$$B = I_{z>0}$$

**/
Layer::RectLinear(output) {
	next.inputs[][] = setbounds(output,0,.NaN);
	B[][] = next.inputs .> 0;
	}

/**SoftMax (Multinomial Logit) Activation.
@param output.  TxN matrix of neuron outputs
z is stored as input in the next layer.
$$z = f(y) = f(b+xA).$$
**/
Layer::SoftMax(output) 	{
	decl ev = exp(output-maxr(output));
	next.inputs[][] = (ev ./ sumr(ev));
	B[][] = next.inputs .* (1-next.inputs);  	//dF = F(1-F)
	}	
