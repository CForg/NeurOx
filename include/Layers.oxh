#include <oxstd.oxh>
#include <oxdraw.oxh>

enum{Ninputs,Nneurons,Ndims}  	  // dimensions of weight matrices

/** @name ActTypes**/
enum{LinAct,RecLinAct,SMAct,SoftAct,NActivationTypes}

struct Layer {
	static 	decl DROPOUT;
			const decl 	
						Dims,
						NW,
						Activation;	//will hold selected activation function
	   	  		decl 	next,		//pointer to next layer
		  	   			prev,		//pointer to previous layer
						inputs,		//input 
						GM,			//gradient of weights in matrix form
						B;			// Back propogation storage
	static	Dimensions(A);
	virtual Forward();
	virtual Backward();
	virtual UpdateWeights(vW);
	virtual Plot();
	 		Linear(output);
			Sigmoid(output);
			RectLinear(output);
			SoftMax(output);	
	}

struct 	Dropout : Layer{
		decl rate;
		decl curdrops;  //masking vector/matrix
				Dropout(rate);
				Forward();
				Backward();
				UpdateWeights(vW);
	}
	
struct Dense : Layer { 
	const decl
					lambda;		 //coefficient on regularization
	decl
					myvW,		 //vectorized weights
					MyW0,		//spot in parameter vector for my weights & biases
					bias,		//bias vector
					weights;   //weight matrix
			Dense(dims,Activation=LinAct,lambda=0.0);
	Plot();
	Forward();
	Backward();
	UpdateWeights(vW);
	}	