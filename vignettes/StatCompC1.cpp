// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <algorithm>  //nth_element()
#include <cmath>      //exp()  log()
#include <iomanip>    //erfc
using namespace Eigen;
using namespace Rcpp;

double pnorm(double value) {
  double M = sqrt(0.5);
  return 0.5 * erfc(-value * M);
}

double A1(int n, Eigen::MatrixXd A) {  // n is the number of sample
  double c = 0;
  for (int i = 0; i <= n - 2; i++) {
    for (int j = i + 1; j < n; j++) {
      c = c + A(i, j) * A(i, j);
    }
  }
  return (c);
}

double A2(int n, Eigen::MatrixXd A) {  // n is the number of sample

  double c = 0;
  for (int i = 0; i <= n - 1; i++) {
    for (int j = 0; j <= n - 2; j++) {
      if (j != i) {
        for (int k = j + 1; k <= n - 1; k++) {
          if (k != i) {
            c = c + A(i, j) * A(i, k);
          }
        }
      }
    }
  }
  return (c);
}

double A3(int n, Eigen::MatrixXd A) {  // n is the number of sample

  double c = 0;
  for (int i = 0; i <= n - 4; i++) {
    for (int j = i + 1; j < n; j++) {
      for (int k = i + 1; k <= n - 2; k++) {
        if (k != j) {
          for (int l = k + 1; l <= n - 1; l++) {
            if (l != j) {
              c = c + A(i, j) * A(k, l);
            }
          }
        }
      }
    }
  }
  return (c);
}

double C1(int n,
          int m,
          Eigen::MatrixXd A) {  // n,m is the number of sample X1,X2

  double c = 0;
  for (int i = 0; i <= n - 1; i++) {
    for (int j = 0; j < m; j++) {
      c = c + A(i, j) * A(i, j);
    }
  }
  return (c);
}

double C2(int n,
          int m,
          Eigen::MatrixXd A) {  // n,m is the number of sample X1,X2

  double c = 0;
  for (int i = 0; i <= n - 1; i++) {
    for (int j = 0; j <= m - 2; j++) {
      for (int k = j + 1; k < m; k++) {
        c = c + A(i, j) * A(i, k);
      }
    }
  }
  return (c);
}

double C3(int n,
          int m,
          Eigen::MatrixXd A) {  // n,m is the number of sample X1,X2
  double c = 0;
  for (int i = 0; i <= n - 2; i++) {
    for (int j = 0; j < m; j++) {
      for (int k = i + 1; k < n; k++) {
        for (int l = 0; l < m; l++) {
          if (l != j) {
            c = c + A(i, j) * A(k, l);
          }
        }
      }
    }
  }
  return (c);
}

//' @title A more effective functions for testing the equality of two covariance matrices which are high-dimension
//' @description use Cpp to speed up the codes in R package 'equalCovs'
//' @param sam1 First sample, it must be array with structure size1*p, p is the dimension of data
//' @param sam2 Second sample, it must be array with structure size2*p, p is the dimension of data
//' @return test statistics and p-values 
//' @examples
//'\dontrun{
//' library(mvtnorm)
//'p<-700 # the dimension of multivariate
//'
//'theta1<-2
//'theta2<-1
//'mat1<-diag(theta1,p-1)
//'mat2<-diag(theta1+theta1*theta2,p-1)
//'mat3<-diag(theta2,p-2)
//'
//'mat1<-rbind(mat1,rep(0,p-1))
//'mat2<-rbind(mat2,rep(0,p-1))
//'mat3<-rbind(mat3,rep(0,p-2),rep(0,p-2))
//'
//'mat1<-cbind(rep(0,p),mat1)
//'mat2<-cbind(rep(0,p),mat2)
//'mat3<-cbind(rep(0,p),rep(0,p),mat3)
//'sigma1<-mat1+t(mat1)+diag(1+theta1^2,p)
//'sigma2<-mat2+t(mat2)+mat3+t(mat3)+diag(1+theta1^2+theta2^2,p)
//'
//'n1<-80 #size1
//'n2<-80 #size2
//'s1<-rmvnorm(size1,runif(p,0,5),sigma1) # generate the samples
//'s2<-rmvnorm(size2,runif(p,-3,3),sigma2)
//'
//' test the result of three functions
//'library(equalCovs)
//'equalCovs(sam1,sam2,size1,size2)
//'equalCovs_C(sam1,sam2)
//'equalCovs_Matrix(sam1, sam2)
//'
//' #test the time of three functions
//'
//'library(microbenchmark)
//'t<-microbenchmark(Cpp=equalCovs_C(s1,s2),Fort=equalCovs(s1,s2,n1,n2),Mat=equalCovs_Matrix(s1,s2))
//'summary(t)[,c(1,3,5,6)]
//' }
//' @export
//[[Rcpp::export]]
Eigen::Vector2d equalCovs_C(Eigen::MatrixXd sam1, Eigen::MatrixXd sam2) {
  double size1, size2;
  double a1, a2, a3;
  double b1, b2, b3;
  double c1, c2, c3, c4;
  double A_n1, B_n2, C_n, T_n;
  double Sd_prime, test_stat, pvalue;
  Eigen::MatrixXd A_mat, B_mat, C_mat1, C_mat2;
  Eigen::Vector2d test;

  size1 = sam1.rows();
  size2 = sam2.rows();
  // obtain the test statistic in (2.1)
  A_mat = sam1 * sam1.adjoint();
  a1 = A1(size1, A_mat);
  a1 = (2 * a1) / (size1 * (size1 - 1));

  a2 = A2(size1, A_mat);
  a2 = (4 * a2) / (size1 * (size1 - 1) * (size1 - 2));

  a3 = A3(size1, A_mat);
  a3 = (8 * a3) / (size1 * (size1 - 1) * (size1 - 2) * (size1 - 3));

  A_n1 = a1 - a2 + a3;

  B_mat = sam2 * sam2.adjoint();
  b1 = A1(size2, B_mat);
  b1 = (2 * b1) / (size2 * (size2 - 1));

  b2 = A2(size2, B_mat);
  b2 = (4 * b2) / (size2 * (size2 - 1) * (size2 - 2));

  b3 = A3(size2, B_mat);
  b3 = (8 * b3) / (size2 * (size2 - 1) * (size2 - 2) * (size2 - 3));

  B_n2 = b1 - b2 + b3;

  // obtain the test statistic in (2.2)
  C_mat1 = sam1 * sam2.adjoint();
  C_mat2 = sam2 * sam1.adjoint();

  c1 = C1(size1, size2, C_mat1);
  c1 = -(2 * c1) / (size1 * size2);

  c2 = C2(size2, size1, C_mat2);
  c2 = (4 * c2) / (size1 * size2 * (size1 - 1));

  c3 = C2(size1, size2, C_mat1);
  c3 = (4 * c3) / (size1 * size2 * (size2 - 1));

  c4 = C3(size1, size2, C_mat1);
  c4 = -(4 * c4) / (size1 * size2 * (size1 - 1) * (size2 - 1));

  C_n = c1 + c2 + c3 + c4;

  // the estimator
  T_n = A_n1 + B_n2 + C_n;

  // the standard deviation
  Sd_prime =
      2 * (1 / size1 + 1 / size2) *
      ((size1 / (size1 + size2)) * A_n1 + (size2 / (size1 + size2)) * B_n2);

  test_stat = T_n / Sd_prime;
  pvalue = 1 - pnorm(test_stat);

  test << test_stat, pvalue;

  return test;
}

double A1Matrix(
    Eigen::MatrixXd X) {  // n is the nrow of X,X is the sample matrix
  Eigen::MatrixXd XXt;    // X*t(X)
  Eigen::MatrixXd XXt2;   // pointwise square of X*t(X),i.e. XXt.^2 in matlab
  double Z = 0;
  XXt = X * X.adjoint();
  XXt2 = XXt.array().square();
  Z = XXt2.sum() - XXt2.trace();
  return Z;
}

double A2Matrix(
    Eigen::MatrixXd X) {  // n is the nrow of X,X is the sample matrix
  Eigen::MatrixXd A;      // X*t(X)
  Eigen::MatrixXd B;      // diag(A)
  Eigen::MatrixXd C;      // A*A
  Eigen::MatrixXd D;      // A.^2
  Eigen::MatrixXd E;      // B*A
  double Z = 0;

  A = X * X.adjoint();
  B = A.diagonal().asDiagonal();
  C = A * A;
  D = A.array().square();
  E = B * A;
  Z = C.sum() - 2 * E.sum() - D.sum() + 2 * D.trace();
  return Z;
}

double A3Matrix(
    Eigen::MatrixXd X) {  // n is the nrow of X,X is the sample matrix
  Eigen::MatrixXd XXt;    // X*t(X)
  Eigen::MatrixXd XXt2;   // pointwise square of X*t(X),i.e. XXt.^2 in matlab
  Eigen::MatrixXd C;
  double Z = 0;
  double A = 0;
  double B = 0;
  XXt = X * X.adjoint();
  XXt2 = XXt.array().square();
  A = XXt.sum();
  B = XXt.trace();
  C = XXt.diagonal().asDiagonal();

  Z = pow(A, 2) - 2 * XXt.trace() * A - 4 * (XXt * XXt).sum() + pow(B, 2) +
      2 * XXt2.sum() + 8 * (C * XXt).sum() - 6 * XXt2.trace();
  return Z;
}

double C1Matrix(
    Eigen::MatrixXd X,
    Eigen::MatrixXd Y) {  // n is the nrow of X,X is the sample matrix
  Eigen::MatrixXd XYt;    // X*t(Y)
  Eigen::MatrixXd XYt2;   // pointwise square of X*t(X),i.e. XXt.^2 in matlab
  double Z = 0;
  XYt = X * Y.adjoint();
  XYt2 = XYt.array().square();
  Z = XYt2.sum();
  return Z;
}

double C2Matrix(
    Eigen::MatrixXd X,
    Eigen::MatrixXd Y) {  // n is the nrow of X,X is the sample matrix
  Eigen::MatrixXd XYt;    // X*t(Y)
  Eigen::MatrixXd XYt2;   // pointwise square of X*t(X),i.e. XXt.^2 in matlab
  Eigen::MatrixXd A;
  double Z = 0;
  XYt = X * Y.adjoint();
  XYt2 = XYt.array().square();
  A = XYt * XYt.adjoint();
  Z = A.sum() - XYt2.sum();
  return Z;
}

double C3Matrix(
    Eigen::MatrixXd X,
    Eigen::MatrixXd Y) {  // n is the nrow of X,X is the sample matrix
  Eigen::MatrixXd XYt;    // X*t(Y)
  Eigen::MatrixXd XYt2;   // pointwise square of X*t(X),i.e. XXt.^2 in matlab
  Eigen::MatrixXd YXt;
  Eigen::MatrixXd A;
  Eigen::MatrixXd B;
  double Z = 0;
  double C = 0;
  XYt = X * Y.adjoint();
  YXt = Y * X.adjoint();
  A = XYt * XYt.adjoint();
  B = YXt * YXt.adjoint();
  XYt2 = XYt.array().square();
  C = XYt.sum();
  Z = pow(C, 2) - B.sum() - A.sum() + XYt2.sum();
  return Z;
}

//' @title A more effective functions for testing the equality of two covariance matrices which are high-dimension
//' @description use Cpp to speed up the codes in R package 'equalCovs'
//' @param sam1 First sample, it must be array with structure size1*p, p is the dimension of data
//' @param sam2 Second sample, it must be array with structure size2*p, p is the dimension of data
//' @return test statistics and p-values
//' @examples
//'\dontrun{
//' library(mvtnorm)
//'p<-700 # the dimension of multivariate
//'
//'theta1<-2
//'theta2<-1
//'mat1<-diag(theta1,p-1)
//'mat2<-diag(theta1+theta1*theta2,p-1)
//'mat3<-diag(theta2,p-2)
//'
//'mat1<-rbind(mat1,rep(0,p-1))
//'mat2<-rbind(mat2,rep(0,p-1))
//'mat3<-rbind(mat3,rep(0,p-2),rep(0,p-2))
//'
//'mat1<-cbind(rep(0,p),mat1)
//'mat2<-cbind(rep(0,p),mat2)
//'mat3<-cbind(rep(0,p),rep(0,p),mat3)
//'sigma1<-mat1+t(mat1)+diag(1+theta1^2,p)
//'sigma2<-mat2+t(mat2)+mat3+t(mat3)+diag(1+theta1^2+theta2^2,p)
//'
//'n1<-80
//'n2<-80
//'s1<-rmvnorm(size1,runif(p,0,5),sigma1) # generate the samples
//'s2<-rmvnorm(size2,runif(p,-3,3),sigma2)
//'
//' test the result of three functions
//'library(equalCovs)
//'equalCovs(sam1,sam2,size1,size2)
//'equalCovs_C(sam1,sam2)
//'equalCovs_Matrix(sam1, sam2)
//'
//' #test the time of three functions
//'
//'library(microbenchmark)
//'t<-microbenchmark(Cpp=equalCovs_C(s1,s2),Fort=equalCovs(sa1,s2,n1,n2),Mat=equalCovs_Matrix(s1,s2))
//'summary(t)[,c(1,3,5,6)]
//' }
//' @export
//[[Rcpp::export]]
Eigen::Vector2d equalCovs_Matrix(Eigen::MatrixXd sam1, Eigen::MatrixXd sam2) {
  double size1, size2;
  double a1, a2, a3;
  double b1, b2, b3;
  double c1, c2, c3, c4;
  double A_n1, B_n2, C_n, T_n;
  double Sd_prime, test_stat, pvalue;
  Eigen::MatrixXd A_mat, B_mat, C_mat1, C_mat2;
  Eigen::Vector2d test;

  size1 = sam1.rows();
  size2 = sam2.rows();

  a1 = A1Matrix(sam1);
  a1 = (1 * a1) / (size1 * (size1 - 1));

  a2 = A2Matrix(sam1);
  a2 = (2 * a2) / (size1 * (size1 - 1) * (size1 - 2));

  a3 = A3Matrix(sam1);
  a3 = (1 * a3) / (size1 * (size1 - 1) * (size1 - 2) * (size1 - 3));

  A_n1 = a1 - a2 + a3;

  b1 = A1Matrix(sam2);
  b1 = (1 * b1) / (size2 * (size2 - 1));

  b2 = A2Matrix(sam2);
  b2 = (2 * b2) / (size2 * (size2 - 1) * (size2 - 2));

  b3 = A3Matrix(sam2);
  b3 = (1 * b3) / (size2 * (size2 - 1) * (size2 - 2) * (size2 - 3));

  B_n2 = b1 - b2 + b3;

  c1 = C1Matrix(sam1, sam2);
  c1 = -(2 * c1) / (size1 * size2);

  c2 = C2Matrix(sam1, sam2);
  c2 = (2 * c2) / (size1 * size2 * (size1 - 1));

  c3 = C2Matrix(sam2, sam1);
  c3 = (2 * c3) / (size1 * size2 * (size2 - 1));

  c4 = C3Matrix(sam1, sam2);
  c4 = -(2 * c4) / (size1 * size2 * (size1 - 1) * (size2 - 1));

  C_n = c1 + c2 + c3 + c4;

  T_n = A_n1 + B_n2 + C_n;

  // the standard deviation
  Sd_prime =
      2 * (1 / size1 + 1 / size2) *
      ((size1 / (size1 + size2)) * A_n1 + (size2 / (size1 + size2)) * B_n2);

  test_stat = T_n / Sd_prime;
  pvalue = 1 - pnorm(test_stat);

  test << test_stat, pvalue;

  return test;
}
