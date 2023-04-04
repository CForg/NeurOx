#import "Layers"
#import <maximize>

enum{BinaryCELoss,CELoss,MSELoss,NLossFunctions}

static decl net;   //object that holds the Network for passing info 

/** Loss as objective that can be used by maximize package **/
Lobj(v,aL,aG,aH);

struct Loss {
	decl
									  vL,
									  loss,
									  B,
		/** networks target. **/      target,
		/** top-leval activations.**/ inputs; 
	virtual value(aL,aGrad);
	Loss(target);
	}

struct CrossEntropy : Loss {
	const decl rng,targcol;
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
			VOLUME; //output level
	decl isbuilt,
		 BatchSize,
		 Nlayers,
		 Nweights,
		 layers,
		 Loss,
		 penalty,
		 floss,
		 grad ;
		 
		 Network();
		 Obj(vW);
		 AddLayers(...args);
		 CrossEntropy();
		 SetBatchAndTarget(LossType,batch,target);
		 Forward();
		 Backward();
		 Output();

	}
	