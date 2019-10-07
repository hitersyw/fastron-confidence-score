#include "mex.h"
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>
#include <fstream>
#include <sstream>
#include "math.h"
#include "Eigen/Array"
#include "Eigen/LU"
#include "Eigen/SVD"
#include "time.h"

USING_PART_OF_NAMESPACE_EIGEN

MatrixXd U;
MatrixXd V;
MatrixXd CC;
MatrixXd CVAinvU;
MatrixXd KTRKinv;

// Sherman-Morrisson Woodbury formula
void update_alpha(const MatrixXd &Ainv, const MatrixXd &KS,
                  const VectorXd &R, const VectorXd &z, 
                  const double lambda, const VectorXd &delta_K, 
                  const VectorXd &delta_Kreg, const VectorXd &delta_kreg, 
                  const int N, const int Q, VectorXd &alpha){

   
    // tmp matrices
    U = 1.0/((double)N) * KS.block(0, 0, N, Q).transpose() * R.asDiagonal() * delta_K + lambda * delta_Kreg;
    V = U.transpose();
    CC = 1.0/((double)N) * delta_K.transpose() * R.asDiagonal() * delta_K + lambda * delta_kreg;
    CVAinvU = (CC - V * Ainv * U).inverse(); 
    
    KTRKinv.block(0, 0, Q, Q) = Ainv + (Ainv * U) * CVAinvU * (V * Ainv);
    KTRKinv.block(0, Q, Q, 1) = -Ainv * U * CVAinvU;
    KTRKinv.block(Q, 0, 1, Q) = -CVAinvU * V * Ainv;
    KTRKinv.block(Q, Q, 1, 1) = CVAinvU;
    
    alpha = KTRKinv * KS.transpose() * R.asDiagonal() * z;
}
        
void mexFunction(
    int nlhs, mxArray* plhs[],
    int nrhs, const mxArray* prhs[])
{
	// declarations
	double* bestN;
    double* fval_bestN;
	int p, c, q, n, col, row;
    double reg, fval, fval_best;
    
    clock_t start, end;
    double cpuTime;


	MatrixXd Phi;
	MatrixXd KS;
	MatrixXd Kreg;
    VectorXd delta_K;
    VectorXd delta_Kreg;
    VectorXd delta_kreg;        
	MatrixXd Y;
	MatrixXd Z;
	MatrixXd R;
	MatrixXd T;
	MatrixXd onesY;
	MatrixXd KRKT;
	MatrixXd dat;
    MatrixXd dat_trans;
    MatrixXd OldAlpha;
	MatrixXd Alpha;
    VectorXd alpha;  
	MatrixXd A;
    VectorXd r;
    VectorXd sum_y;
    MatrixXd Ainv;
        
    // dimensions
    const unsigned long int N = mxGetN(prhs[0]); // number of all points
    const unsigned long int D = mxGetM(prhs[0]); // dimension of features
    const unsigned long int Q = mxGetN(prhs[1]); // size of the subset
    const unsigned long int C = mxGetN(prhs[3]); // number of classes
    const unsigned long int P = mxGetN(prhs[5]); // number of points to be tested
    
    // resize
    U.resize(Q, 1);
    V.resize(1, Q);
    CC.resize(1, 1);
    KTRKinv.resize(Q+1, Q+1);
    
    // input
	double *mxPhi    = mxGetPr(prhs[0]); // feature matrix
	double *mxKS     = mxGetPr(prhs[1]); // subset kernel matrix
	double *mxKreg   = mxGetPr(prhs[2]); // regularization kernel matrix
	double *mxY      = mxGetPr(prhs[3]); // probabilities
	double *mxZ      = mxGetPr(prhs[4]); // z
	double *points   = mxGetPr(prhs[5]); // points to be tested
	double *subset   = mxGetPr(prhs[6]); // subset
	double *mxlambda = mxGetPr(prhs[7]);  // regularization parameter
	double lambda    = mxlambda[0];
    double *mxsigma  = mxGetPr(prhs[8]);  // kernel parameter
	double sigma     = mxsigma[0];
	double *mxtargets  = mxGetPr(prhs[9]); // targets
	double *mxAinv   = mxGetPr(prhs[10]); // Hessian
    double *mxOldAlpha   = mxGetPr(prhs[11]); // old parameters
    double *mxtempInt  = mxGetPr(prhs[12]); // temporal integration
    double tempInt     = mxtempInt[0];  // temporal integration
    
    if (nrhs < 14){
        fval_best = 1e16;
    }
    else{
       // double *mxfval_best = mxGetPr(prhs[10]);
        fval_best   = mxGetScalar(prhs[13]);
    }
    
	if (mxGetM(prhs[3]) != mxGetM(prhs[9])){
		mexErrMsgTxt("targets and probabilities do not have the same dimensions.");
	}
    
    // output
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    bestN = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    fval_bestN = mxGetPr(plhs[1]);
    
	// convert subset kernel matrix KS to MatrixXD	
	KS.resize(N, Q + 1);
	for (col = 0; col < Q; ++col) {
		for (row = 0; row < N; ++row) {
			KS(row, col) = mxKS[(col * N) + row];
		}
	}
	
	// convert kernel matrix K to MatrixXD	
	Phi.resize(D, N);
	for (col = 0; col < N; ++col) {
		for (row = 0; row < D; ++row) {
			Phi(row, col) = mxPhi[(col * D) + row];
		}
	}
        
	// convert regularization kernel matrix KQ to MatrixXD
	Kreg.resize(Q + 1, Q + 1);
	for (col = 0; col < Q; ++col) {
		for (row = 0; row < Q; ++row) {
			Kreg(row, col) = mxKreg[(col * Q) + row];
		}
	}
	
    // convert old inverse
    Ainv.resize(Q * C, Q);
	for (col = 0; col < Q; ++col) {
		for (row = 0; row < (Q * C); ++row) {
			Ainv(row, col) = mxAinv[(col * (Q * C)) + row];
		}
	}
    
    // convert old parameters
    OldAlpha.resize(Q+1, C);
	for (col = 0; col < C; ++col) {
		for (row = 0; row < (Q + 1); ++row) {
			OldAlpha(row, col) = mxOldAlpha[(col * (Q + 1)) + row];
		}
	}
    
	// convert Y and Z to MatrixXD and VectorXD	
	Y.resize(N, C);
	Z.resize(N, C);
	for (c = 0; c < C; ++c) {
		for (row = 0; row < N; ++row) {
			Y(row, c) = mxY[(c * N) + row];
			Z(row, c) = mxZ[(c * N) + row];
		}
	}
	
	// convert Y and Z to MatrixXD and VectorXD	
	T.resize(N, C);
	for (col = 0; col < C; ++col) {
		for (row = 0; row < N; ++row) {
			T(row, col) = mxtargets[(col * N) + row];
		}
	}
    
	// compute probability weights
    R.resize(N, C);
    onesY.setOnes(N, C);
    R = Y.cwise() * (onesY - Y);
        
    // test each points to be in the subset
    r.resize(N);
    KRKT.resize(Q + 1, Q + 1);
    dat.resize(Q + 1, N);
    dat_trans.resize(N, Q + 1);
    Alpha.resize(Q + 1, C);
    alpha.resize(Q + 1, 1);
    A.resize(C, N);
    sum_y.resize(N);
    delta_kreg.resize(1);
    bestN[0] = 0;
    fval_bestN[0] = fval_best;
    delta_Kreg.resize(Q);
    delta_K.resize(N, 1);
    delta_kreg(0) = 1.0;
    for (p = 0; p < P; ++p) {
        sum_y.setZero(N);
        
		// for alpha update: get new kernel matrix rows
        for (int colK1 = 0; colK1 < N; ++colK1) {
             delta_K(colK1) = exp(-(Phi.col(colK1) - Phi.col((int) points[p] - 1)).squaredNorm() / (2.0 * sigma * sigma));
        }

	    // for alpha update: get parts of regularization matrix
	    for (q = 0; q < Q; ++q) {
			delta_Kreg(q) = exp(-(Phi.col((int) subset[q] - 1) - Phi.col((int) points[p] - 1)).squaredNorm() / (2.0 * sigma * sigma));
		}

        // get subset kernel matrix
		KS.col(Q) = delta_K;	

	    // get regularization matrix
	    for (q = 0; q < Q; ++q) {
			Kreg(Q, q) = exp(-(Phi.col((int) points[p] - 1) - Phi.col((int) subset[q] - 1)).squaredNorm() / (2.0 * sigma * sigma));
			Kreg(q, Q) = exp(-(Phi.col((int) subset[q] - 1) - Phi.col((int) points[p] - 1)).squaredNorm() / (2.0 * sigma * sigma));
		}
        Kreg(Q, Q) = 1.0;

        // compute paramters for all classes
		for (int c = 0; c < C; ++c){
            
            // parameters
            if (Q == 0){
                Alpha.block(0, c, Q + 1, 1) = (1.0 / N * KS.transpose() * R.col(c).asDiagonal()
                * KS + lambda * Kreg).inverse() * KS.transpose() * R.col(c).asDiagonal() * Z.col(c);
            }
            else{
                update_alpha(Ainv.block(c * Q, 0, Q, Q), KS, R.col(c), Z.col(c), lambda, delta_K, 
                             delta_Kreg, delta_kreg, N, Q, alpha);
                Alpha.block(0, c, Q + 1, 1) = alpha;  
            }

            // new activities
            A.block(c, 0, 1, N) = (Alpha.col(c).transpose() * KS.transpose()).cwise().exp();
			
            sum_y = sum_y + A.row(c).transpose();
		}
        
		// normalize activities yielding probabilities
        for (int c = 0; c < C; ++c){
            Y.col(c) = A.row(c).cwise() / sum_y.transpose();
        }
        
        // set max and min value of probabilities
        for (int n = 0; n < N; ++n){
			for (int c = 0; c < C; ++c){
				if (Y(n, c) < 1e-8){
					Y(n, c) = 1e-8;
				}
				if (Y(n, c) > (1.0 - 1e-12)){
					Y(n, c) = 1.0 - 1e-12;
				}
			}
		}

        // regularization value
        reg = 0.0;
        for (int c = 0; c < C; ++c){
            reg = reg + lambda / 2.0 * (Alpha.col(c).transpose() * Kreg * Alpha.col(c)).trace();
        }
        
        // objective function value
        fval = -1.0 / N * (T.cwise() * Y.cwise().log()).sum() + reg;

        if (fval_best >= fval) {
			bestN[0] = points[p];
            fval_bestN[0] = fval;
            fval_best = fval;
        }
	}
}
