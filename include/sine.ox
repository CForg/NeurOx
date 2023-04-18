/**	Ox version of https://cs231n.github.io/neural-networks-case-study/
	translated by CF	
**/
#include <oxstd.oxh>
#include <oxdraw.oxh>
#include <oxfloat.oxh>

/**	Generate and Graph sine wave
@param N number of points per spiral
@return 

@comments

**/
sine(N=1000) {  
  decl y,X,tag = "sine_"+"_"+sprint(N);  
  X=range(0,1,1/(N-1))';  
  y = sin( M_2PI  * X); 
	DrawXMatrix(0,y',"",X',"",1,2);
  DrawAxisAuto(0,0,1,ANCHOR_USER, 0.0, 0.0);  
  SaveDrawWindow(tag+".pdf");
  savemat(tag+".dta",X);
  return {y,X};
}
