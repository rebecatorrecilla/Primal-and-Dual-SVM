param m, integer;
param m_test, integer;
param n, integer;

param nu;

param A_train {1..m,1..n};
param y_train {1..m};

param y_test {1..m_test};
param A_test {1..m_test,1..n};

param sigma := sqrt(n/2);

var lambda {1..m} >=0, <=nu;


maximize SVM_kernel:
	(sum{i in 1..m} lambda[i]) - (1/2)*(sum{i in 1..m}(sum{j in 1..m}lambda[i]*y_train[i]*lambda[j]*y_train[j]*exp(-(sum{k in 1..n}(A_train[i,k]-A_train[j,k])^2)/(2*sigma^2))));
	
subject to separation:
	sum{i in 1..m}lambda[i]*y_train[i] = 0;