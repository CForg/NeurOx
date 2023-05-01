#include <oxstd.oxh>
#import <packages/oxurl/oxurl>
#import <packages/Python/Python>
getpip();
trynumpy();

main() {
    /* Uncomment this call to get pip and then use it to install numpy   */
    //    getpip();
      trynumpy();
}

getpip() {
	decl url = new OxUrl();
	decl res = url.Save("get-pip.py", "https://bootstrap.pypa.io/get-pip.py");

	if (res != CURLE_OK) 
        oxrunerror("download fail");
	delete url;
    println(systemcall("python get-pip.py"));
    println(systemcall("pip install numpy"));
    println(systemcall("pip install opencv-python"));
    }

trynumpy() {
	decl py = new Python();

	println("\nTry some NumPy");
	// get the value of numpy.version.version
	println("NumPy version:    ", py.get("numpy", "version.version"));

	// write some numpy Python code, passing matrices back to Ox:
	decl numpy_code =
`
import numpy

def ranu_numpy():
	x = numpy.random.uniform(size=3);
	return x;
def test1_numpy():
	x = numpy.array(((3, 6, 7), (5, 2, 0)));
	return x;
def test2_numpy():
	x = numpy.array([[3.3, 6, 7], [5, 2, 0]]);
	return x;
def test3_numpy():
	print("Calling test3_numpy, matrix x:");
	x = numpy.matrix(((3.3, 6, 7), (5, 2, 0)));
	print(x);
	return x;
`;

	// no need to import numpy, because that is done by default
	//	py.import("numpy");
	println(py.str(py.Module("numpy")));
	
	// import the code in Python, as module "ox"
	py.import("ox",	numpy_code);

	// now make some calls and print the results in Ox
	println("uniform rng from NumPy:    ", "%v", py.call("ox", "ranu_numpy"));
	println("array of ints from NumPy:  ", "%v", py.call("ox", "test1_numpy"));
	println("array of dbls from NumPy:  ", "%v", py.call("ox", "test2_numpy"));
	println("matrix from NumPy:         ", "%v", py.call("ox", "test3_numpy"));
    }

