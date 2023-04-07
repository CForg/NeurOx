#import "Layers"
#import <maximize>

enum{NoLoss,BinaryCELoss,CELoss,MSELoss,NLossFunctions}

static decl net;   //object that holds the Network for passing info 

/** Loss as objective that can be used by maximize package **/
Lobj(v,aL,aG,aH);

struct Loss {
	decl
									  vL,
									  loss,
									  B,
		/** networks target. **/      target,
		/** top-leval activations.**/ inputs,
		 							  prediction,
		 							  accuracy;
			Loss(target);
  	virtual value();
	}

struct CrossEntropy : Loss {
	const decl rng, targcol, J;
	virtual value();
	CrossEntropy(target);
	}
	
struct BinaryCrossEntropy : CrossEntropy {
	value();
	BinaryCrossEntropy(target);
	}	
struct MeanSquareError : Loss {
	value();
	MeanSquareError(target);
	}

struct Network  {
	static decl
			BACKPROPAGATION, // use Backward for gradient (otherwise numeric)
			PREDICTING,  	// compute prediction and accuracy
			VOLUME; 		//output level
	decl isbuilt,
		 BatchSize,
		 Nlayers,
		 Nparams,
		 layers,
		 Loss,
		 penalty,
		 floss,
		 grad,
		 vLabels ;
		 
		 Network();
		 Obj(vW);
		 AddLayers(...args);
		 CrossEntropy();
		 SetParameters(vW);
		 SetBatchAndTarget(LossType,batch,target=0);
		 Forward();
		 Backward();
		 Output();

	}
	