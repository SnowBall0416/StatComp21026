#include <Rcpp.h>
using namespace Rcpp;

//' @title A Gibbs sampler using Rcpp
//' @description A Gibbs sampler using Rcpp
//' @param n x will obey B(n,y) when y is fixed
//' @param a y will obey Beta(x+a,n-x+b) when x is fixed
//' @param b y will obey Beta(x+a,n-x+b) when x is fixed
//' @return  a random sample matrix
//' @examples
//' \dontrun{
//' n <- 10
//' a <- 5
//' b <- 6
//' Gibbs_C <- GibbsC(n,a,b)
//' }
//' @export
// [[Rcpp::export]]
NumericMatrix GibbsC(int n, int a, int b) {
  int N = 5000;
  int x;
  double y;
  NumericMatrix mat(N, 2);
  mat(0, 0) = 1;
  mat(0, 1) = 0.5;
  for (int i = 1; i < N; i++) {
    y = mat(i - 1, 1);
    mat(i, 0) = rbinom(1, n, y)[0];
    x = mat(i, 0);
    mat(i, 1) = rbeta(1, x + a, n - x + b)[0];
  }
  return (mat);
}