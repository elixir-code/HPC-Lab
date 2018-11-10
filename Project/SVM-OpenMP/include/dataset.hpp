#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
using namespace std;

/* An classification dataset with exclusively numerical features */
template <typename T>
class Dataset
{
public:	
	vector<int> target;
	vector<vector<T> > data;

	int n_data;
	int n_features;
};

/* Read a classification dataset from a libsvm file format */
template <typename T>
Dataset<T> readLibsvmDataset(const char *filename, int n_data, int n_features);

#endif