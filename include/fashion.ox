/**	Ox version of https://cs231n.github.io/neural-networks-case-study/
	translated by CF	
**/
#include <oxstd.oxh>
#import <packages/Python/Python>

const decl fashion_mnist_labels = {"T-shirt/top","Trouser", "Pullover", "Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"};

const decl icode =
`
import cv2
def vectorpng(fn):
    image_data = cv2.imread(fn,cv2.IMREAD_UNCHANGED);
    return image_data;
`;

fashion(dfolder) {  
  decl y,X,tag,newx,classes,py,intc,f,files,k;  
  py = new Python();
  py.import("ox",	icode);
  chdir(dfolder);
  classes = getfolders("*");
  foreach(c in classes)  {
    chdir(c);
    sscan(c,"%u",&intc);
    files =  getfiles("*.png");
    y = X = <>;
    tag = dfolder+"_"+c+"_";
    println("\n Working in: ",getcwd()," ",c," ",intc," ",tag);
    foreach(f in files[k]) {
      newx = py.call("ox", "vectorpng",f);
      X |= vec(newx)';
      y |= intc;
      if (!imod(k,100)) print(".");
      }
    chdir("..");
    savemat(tag+"_y.zip",y);
    savemat(tag+"_X.zip",X);
    }
  chdir("..");      // go back home!
}


get_fashion(dfolder) {  
  decl y,X,tag,newx,newy,k;  
  chdir(dfolder);
  y = X = <>;
  for (k=0;k<10;++k) {
    tag = dfolder+"_"+sprint(k)+"__";
    newx = loadmat(tag+"X.zip");
    newy = loadmat(tag+"y.zip");
    X |= newx;
    y |= newy;
    }
 chdir("..");
 return {y,X} ;
}
