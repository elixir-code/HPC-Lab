#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <vector>
using namespace std;

#include "dataset.hpp"

/* Classifier base class from which all classifiers derive */
template <typename T>
class Classifier
{
	// <define the parameters of the derived classifiers here>

public:
	// Fit the model with the dataset to learn parameters
	virtual void fit(const Dataset<T>& dataset) = 0;

	// Predict the class label for test data
	virtual int predict(const vector<T>& data) = 0;

	// Predict the class labels for test dataset
	vector<int> predict(const Dataset<T>& dataset);
};

/* Support Vector Machine Classification Model */
template <typename T>
class SVC: public Classifier<T>
{
	// Parameters of the SVC classifier: C, tol, Kernel function
	double C, tol, (*kernelFunction)(vector<T>, vector<T>), eps;
	
	// Dataset with which model was fitted (used in prediction)
	Dataset<T> dataset;

	// Parameters to be learnt from dataset: alphas[n_data], b
	double *alphas, b;
	
	// error between prediction f(x) and actual value for examples in training data
	double *errors;

	// Helper functions to find alpha i (given alpha j) and update alpha i, alpha j pair
	int findUpdateAlphaPair(int alpha2_index);
	int updateAlphaPair(int alpha1_index, int alpha2_index);

	// Syncronization variables to indicate that alpha i that can make positive progress was found
	int valid_alpha1_found;

public:

	// Initialize the parameters for SVM Classifier
	SVC(double C, double tol, double eps, double (*kernelFunction)(vector<T>, vector<T>));
	
	void fit(const Dataset<T>& dataset);
	int predict(const vector<T>& data);

	using Classifier<T>::predict;
};

#endif