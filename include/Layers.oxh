#include <oxstd.oxh>

enum{Ninputs,Nneurons,Ndims}  	  // dimensions of weight matrices
enum{LinAct,RecLinAct,SMAct,SoftAct,NActivationTypes}

struct Layer {
	const decl 	Dims,
				NW,
				Activation;	//will hold selected activation function
	   	  decl 	next,		//next layer
		  	   	prev,		//previous layer
				inputs,		//input 
				output,		//output  
				GM,			//gradient in matrix form
				B;			//current Back propogation storage
	static	Dimensions(A);
	virtual Forward();
	virtual Backward();
	virtual UpdateWeights(vW);
	 		Linear();
			Sigmoid();
			RectLinear();
			SoftMax();	
	}

struct 	LayerDropout : Layer{
	static decl DROPOUT;
	const decl rate;
		  decl curdrops;
	LayerDropout(rate);
	Forward();
	Backward();
	UpdateWeights(vW);
	}
	
struct LayerDense : Layer { 
	const decl
		lambda;		 //coefficient on regularization
	decl
		myvW,		 //vectorized weights
		MyW0,		//spot in parameter vector for my weights & biases
		bias,		//bias vector
		weights;
	LayerDense(dims,Activation=LinAct,lambda=0.0);
	Forward();
	Backward();
	UpdateWeights(vW);
	}	