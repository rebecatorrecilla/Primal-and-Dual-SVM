param m, integer; # size training dataset
param m_test, integer; # size test dataset
param n, integer; # number of features

param nu; # regularization parameter

param A_train {1..m,1..n}; # data matrix
param y_train {1..m}; # classification of data

param y_test {1..m_test};
param A_test {1..m_test,1..n};

var gamma;
var w {1..n};
var s{1..m}>=0;


minimize SVM_primal:
	(1/2) * (sum {j in 1..n} w[j]^2) + nu*(sum{i in 1..m} s[i]);
	
subject to separation {i in 1..m}:
	y_train[i]*((sum{j in 1..n} w[j]*A_train[i,j]) + gamma) + s[i]>=1;