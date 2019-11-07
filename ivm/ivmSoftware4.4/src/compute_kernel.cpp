#include "mex.h"
#include "math.h"
#include "stdio.h"
#include "Eigen/Array"
#include "Eigen/LU"
#include "Eigen/SVD"

USING_PART_OF_NAMESPACE_EIGEN

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{

if (nrhs != 3)
      mexErrMsgTxt("Incorrect number of input arguments");
if (nlhs != 1)
      mexErrMsgTxt("Incorrect number of output arguments");
   
/* kernel parameter */
double *mxsigma = mxGetPr(prhs[2]);
double sigma = mxsigma[0];

/* read phi and convert to MatrixXD */	
double *data1 = mxGetPr(prhs[0]);
int D = mxGetM(prhs[0]);
int N = mxGetN(prhs[0]);
MatrixXd phi;

phi.resize(D, N);
for (int col1 = 0; col1 < N; ++col1) {
	for (int row1 = 0; row1 < D; ++row1) {
		phi(row1, col1) = data1[(col1 * D) + row1];
	}
}

/* read phiq and convert to MatrixXD */	  
double *data2 = mxGetPr(prhs[1]);
int N2 = mxGetM(prhs[1]);
int S = mxGetN(prhs[1]);
MatrixXd phiQ;
		
phiQ.resize(D, S);
for (int col2 = 0; col2 < S; ++col2) {
	for (int row2 = 0; row2 < D; ++row2) {
		phiQ(row2, col2) = data2[(col2 * D) + row2];
	}
}

/* compute kernel matrix */
MatrixXd Kernel;
Kernel.resize(N, S); 

for (int colK2 = 0; colK2 < S; ++colK2) {
	for (int colK1 = 0; colK1 < N; ++colK1) {
		 Kernel(colK1, colK2) = exp(-(phi.col(colK1) - phiQ.col(colK2)).squaredNorm() / (2.0 * sigma * sigma));
		 // Kernel(colK1, colK2) = sqrt(1 + (phi.col(colK1) - phiQ.col(colK2)).squaredNorm() / (2.0 * sigma * sigma));
	}
}

/* Create an mxArray for the output data */
plhs[0] = mxCreateDoubleMatrix(N, S, mxREAL);
double* K       = mxGetPr(plhs[0]);

for (int colK2 = 0; colK2 < S; ++colK2) {
	for (int rowK2 = 0; rowK2 < N; ++rowK2) {
		 K[(colK2 * N) + rowK2] = Kernel(rowK2, colK2);
	}
}

}
