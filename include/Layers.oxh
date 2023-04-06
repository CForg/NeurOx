#include <oxstd.oxh>
#include <oxdraw.oxh>

/** @name Dimensions **/
enum{Ninputs,Nneurons,Ndims}  	 

/** Tags for integer codes for types of Activations.   @name ActTypes**/
enum{LinAct,RecLinAct,SMAct,SoftAct,NActivationTypes}

/** Represent a layer of neurons in a network. **/
struct Layer {
	static 	decl DROPOUT;	//**If TRUE any dropout layer will be activated
			const decl 	
						/** will hold selected activation function**/
						Dims,
						NW,
						Activation;	
				decl    next,		//** pointer to next layer
		  	   			prev,		//** pointer to previous layer
						inputs,		//input 
						GM,			//gradient of weights in matrix form
						B;			// Back propogation storage
	static	Dimensions(A);
	virtual Forward();
	virtual Backward();
	virtual SetParameters(vW);
	virtual Plot();
	 		Linear(output);
			Sigmoid(output);
			RectLinear(output);
			SoftMax(output);	
	}

/** Represent a Droput Layer of Neurons in a Network. **/
struct 	Dropout : Layer{
		decl rate;
		decl curdrops;  //masking vector/matrix
				Dropout(rate);
				Forward();
				Backward();
				SetParameters(vW);
	}
	
/** Represent a Dense  Layer of Neurons in a Network. **/
struct Dense : Layer { 
	const decl
					lambda;		 //** coefficient on regularization
	decl
					myvW,		 //** vectorized weights
					MyW0,		//** spot in parameter vector for my weights & biases
					bias,		//** bias vector
					weights;   //** weight matrix
			Dense(dims,Activation=LinAct,lambda=0.0,ibias=0,iweights=0);
	Plot();
	Forward();
	Backward();
	SetParameters(vW);
	}	