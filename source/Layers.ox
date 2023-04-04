#include "Layers.oxh"

Layer::Linear()		{
	next.inputs[][] = output;	//just pass output 
	B[][] = 1.0;
	}
Layer::Sigmoid() 	{
	next.inputs[][] =  1.0 ./(1+exp(-output));
	B[][] = next.inputs.*(1-next.inputs);		
	}
Layer::RectLinear() {
	next.inputs[][] = setbounds(output,0,.NaN);
	B[][] = next.inputs.>0;
	}
Layer::SoftMax() 	{
	decl ev = exp(output-maxr(output));
	next.inputs[][] = ev ./ sumr(ev);
	B[][] = next.inputs .* (1-next.inputs);  	//probably wrong!
	}
	
Layer::Dimensions(A)   { return rows(A)~columns(A); }

/** Create a dense layer in a neural network.
@param dims  <Ninputs,Nneurons> dimensions of the layer
@param ActType integer code for the Activation
@lambda absolute value regularization factor [default=0.0]
**/
LayerDense::LayerDense(Dims,ActType,lambda) {
	this.Dims = Dims;
	this.lambda = lambda;
	bias = -1 + 2*ranu(1,Dims[Nneurons]);
	weights = -1 + 2*ranu(Dims[Ninputs],Dims[Nneurons]);
	NW = int(Dims[Nneurons]+prodr(Dims));//weights & bias parameters
	myvW = zeros(NW,1);
	GM = zeros(weights)|0;  // matrix for bias and weight gradients
	switch_single (ActType) {
		case LinAct 		:  Activation = Linear;
		case RecLinAct		:  Activation = RectLinear;
		case SMAct			:  Activation = Sigmoid;
		case SoftAct		:  Activation = SoftMax;
		default				:  oxrunerror("Activation Type Invalid");
		}
    }

LayerDense::UpdateWeights(newVW) {
	bias[] = newVW[MyW0:MyW0+Dims[Nneurons]-1];
	myvW[] = newVW[(MyW0+Dims[Nneurons]):(MyW0+NW-1)];
	weights[][] = reshape(myvW,Dims[Ninputs],Dims[Nneurons]);
	return  lambda>0.0   //only compute norm() if necessary
			? lambda*(norm(myvW,1)+norm(bias,1))
			: 0.0;
	}
	
/** Forward propogation.

**/
LayerDense::Forward() {
    output[][] = bias + inputs*weights;	
	Activation();
	}

LayerDense::Backward() {
	//aggregate over inputs, outer product of inputs and forward gradients
	decl t;
	for (GM[][] = 0.0, t=0 ;t<rows(inputs);++t) 
		GM += (1~inputs[t][])'*B[t][];	 
	if (isclass(prev)) // not the first layer so
		prev.B[][] .*= B*weights';		//continue chain rule, multiply by weights' & previous layer's activation
	return vecr(GM);   //need to confirm weights line up with reshape()
	}

LayerDropout::LayerDropout(rate){
	this.rate = rate;
	DROPOUT = TRUE;
	NW = 0;  // no weights in  a dropout layer
	}
LayerDropout::UpdateWeights(newvW) {
	return 0.0;   
	}
LayerDropout::Forward() {
/*	next.inputs[] = inputs;
	B[][] = 1.0;
	return;
*/
	curdrops[][] = (DROPOUT) ?
							ranu(rows(curdrops),columns(curdrops)).>rate  //1,columns(curdrops)
							: 1.0;	
	next.inputs[][] = output[][] = curdrops .* inputs / (1-DROPOUT*rate);
//	println("D ",DROPOUT,curdrops,inputs[:10][:10],next.inputs[:10][:10]);
	B[][] = (next.inputs.>0)/(1-DROPOUT*rate);
	}
LayerDropout::Backward() {
//	println("**",prev.B[:20][:10],B[:20][:10]);
	prev.B[][] .*= B;	// multiply my activation gradient x future effects
	}