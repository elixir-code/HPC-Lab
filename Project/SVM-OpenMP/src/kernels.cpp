#include "kernels.hpp"

/* All Kernels assume that both vectors have same length */


// Perform the for product of two feature vectors
template <typename T>
double dotProduct(vector<T> feature_vector1, vector<T> feature_vector2)
{
	int i;

	double dot_product = 0;
	
	// #pragma omp parallel for private(i) reduction(+:dot_product)
	for(i=0; i<(int)feature_vector1.size(); ++i)
		dot_product += feature_vector1[i] * feature_vector2[i];
	
	return dot_product;
}

template double dotProduct(vector<int>, vector<int>);
template double dotProduct(vector<float>, vector<float>);
template double dotProduct(vector<double>, vector<double>);