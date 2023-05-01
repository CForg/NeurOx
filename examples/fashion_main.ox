#include <oxstd.oxh>
#include "../include/fashion.ox"   // the fashion data generating functions

main() {
  	fopen("convert_fashion_data","l");
    fashion("train");    
	fashion("test");
	fclose("l");
	}