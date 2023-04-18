/* This replicates the code on Page 344 of NNFS usng NeurOx */
#import "NeurOx"        //import the NeurOx package: includes Layers,Network, Optimize
#include "spiral.ox"   // the spiral generating function

enum{Nk=100,Nn=2,K=3,Nh=64}   
main() {
    decl net,batch,target,layer0,layer1,W,opt;

    [target,batch] = spiral(Nk,K);         

    layer0 =       zeros(1,Nh)
            | 0.01*rann(Nn,Nh);             //stack weights under bias
    layer1 =      zeros(1,K)
            |0.01*rann(Nh,K);   
    W = vecr(layer0)|vecr(layer1);          //vectorize all parameters (will be reshaped interally)

    net = new Network(CELoss);                                      // set Loss as Cross Entropy (multinomial logit));

	net.AddLayers(
        new Dense( <Nn,Nh>, RecLinAct, 5E-4 ),      //add the RecLinAct layer. Set lambda to 5E-4
        new Dense( <Nh,K>,  SoftAct )              // default lambda = 0.0 only 2 arguments
        );
	net.SetBatchAndTarget(batch,target);
    net.SetParameters(W); 
    //   opt = new BFGS(net);     MaxControl(1000,20); 
    opt = new SGD(net,1.0,1E-3,0.9);
    opt.itmax = 10000;
    opt.iterate(&W);
    net.PREDICTING = TRUE;
    net.Obj(W);
    println("In sample Accuracy:",net.Loss.accuracy," Penalty: ",net.penalty);

    /*  Create an out-of-sample data set to validate */
    [target,batch] = spiral(Nk,K);         
    net.SetBatchAndTarget(batch,target);
    net.PREDICTING = TRUE;
    net.Obj(W);
    println("Accuracy Out of Sample:",net.Loss.accuracy," Penalty: ",net.penalty);

    }

/* SHould produce this output.  

Ox 9.06 (Windows_64/Parallel) (C) J.A. Doornik, 1994-2022 (oxlang.dev)
Layers 1. Total parmams: 192
Layers 2. Total parmams: 387
Warning in SetBatchAndTarget: rows of target not equal to neurons at top layer
0 329.562 1 2.22347 0
100 226.763 0.904883 32.134 0.203126
200 206.122 0.818813 291.548 0.315222
300 158.517 0.740929 190.204 0.334897
400 123.506 0.670454 26.4806 0.0557847
500 120.095 0.606682 152.62 0.16532
600 108.204 0.548976 58.4367 0.053912
700 104.308 0.496759 34.125 0.035468
800 101.803 0.449509 54.837 0.0432595
900 98.5325 0.406753 70.9214 0.0507623
1000 94.7921 0.368063 93.4759 0.0559604
1100 94.8866 0.333054 160.395 0.0873814
1200 89.0784 0.301375 17.1326 0.0147984
1300 87.4053 0.272709 23.8122 0.0150935
1400 85.9201 0.24677 29.8525 0.0152237
1500 84.6435 0.223297 7.4889 0.00851972
1600 81.9383 0.202058 12.848 0.0301683
1700 77.3698 0.182839 7.50934 0.0118663
1800 75.3161 0.165448 4.96632 0.00872796
1900 73.4711 0.149711 5.40453 0.00808949
2000 72.1663 0.135471 6.68883 0.00756341
2100 70.9754 0.122585 4.38817 0.00580374
2200 70.1693 0.110925 3.7237 0.00471012
2300 69.5463 0.100374 4.88888 0.0040482
2400 68.9939 0.0908268 4.5189 0.00404828
2500 68.5244 0.0821876 3.53792 0.00311859
2600 68.163 0.0743701 4.73053 0.00292647
2700 67.8603 0.0672962 4.38857 0.00248204
2800 67.605 0.0608952 5.60754 0.00209819
2900 67.384 0.055103 4.14617 0.00188463
3000 67.1931 0.0498618 3.80516 0.00164373
3100 67.0275 0.045119 3.78702 0.00144954
3200 66.8833 0.0408274 3.89809 0.00128489
3300 66.7548 0.036944 3.92462 0.00113822
3400 66.6446 0.03343 3.31043 0.00106255
3500 66.5467 0.0302502 3.92622 0.00105319
3600 66.4585 0.0273729 3.93935 0.000836993
3700 66.3825 0.0247693 3.7244 0.000814236
3800 66.3102 0.0224133 3.83951 0.000697854
3900 66.2412 0.0202814 3.95852 0.000721955
4000 66.1766 0.0183523 4.55006 0.000578457
4100 66.1233 0.0166067 2.97218 0.000500728
4200 66.0783 0.0150271 3.51288 0.000487352
4300 66.0382 0.0135977 4.51671 0.000447917
4400 66.002 0.0123044 4.04954 0.000361542
4500 65.9697 0.011134 2.81307 0.000314371
4600 65.9403 0.010075 3.78811 0.000301788
4700 65.9145 0.00911666 3.9078 0.00027775
4800 65.8908 0.00824951 3.84679 0.00025074
4900 65.8698 0.00746484 3.90566 0.000209667
5000 65.851 0.0067548 3.88974 0.000188541
5100 65.834 0.0061123 3.98755 0.000182499
5200 65.8187 0.00553092 3.34907 0.000153895
5300 65.8048 0.00500483 3.97331 0.0001468
5400 65.7925 0.00452878 3.86175 0.000148115
5500 65.7811 0.00409802 2.99889 0.00012299
5600 65.7711 0.00370823 3.29339 0.000103925
5700 65.7618 0.00335551 3.75805 0.000105364
5800 65.7535 0.00303634 2.92821 9.04762e-05
5900 65.7462 0.00274753 2.98361 8.17863e-05
6000 65.7393 0.00248619 3.79713 7.84475e-05
6100 65.7332 0.00224971 3.18665 6.54083e-05
6200 65.7278 0.00203573 3.72574 5.56486e-05
6300 65.7227 0.00184209 3.76942 5.75942e-05
6400 65.7182 0.00166688 4.35985 4.96813e-05
6500 65.7142 0.00150833 3.73982 4.16417e-05
6600 65.7105 0.00136486 3.80998 4.04261e-05

Iteration complete
In sample Accuracy:0.9 Penalty: 0.870219
Warning in SetBatchAndTarget: Network already built: resizing input and output
Warning in SetBatchAndTarget: rows of target not equal to neurons at top layer
Accuracy Out of Sample:0.886667 Penalty: 0.870219

*/