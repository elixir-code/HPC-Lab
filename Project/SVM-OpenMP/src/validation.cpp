#include "validation.hpp"

#include <cassert>

// Assumption: Number of elements in target and predictions are the same
double computeAccuracy(const vector<int>& target, const vector<int>& predictions)
{
	// Assert that the target and prediction vector have the same size
	assert(target.size() == predictions.size());

	int n_correct_preds = 0;

	int i;

	#pragma omp parallel for private(i) reduction(+:n_correct_preds)
	for(i=0; i<(int)target.size(); ++i)
		if(target[i] == predictions[i])
			n_correct_preds++;

	double accuracy = (double)n_correct_preds/target.size();
	return accuracy;
}


// Assumption: Number of elements in target and predictions are the same
double computePrecision(const vector<int>& target, const vector<int>& predictions)
{
	// Assert that the target and prediction vector have the same size
	assert(target.size() == predictions.size());

	int i;
	int n_pred_positive = 0, n_true_positive = 0;

	#pragma omp parallel for private(i) reduction(+:n_pred_positive, n_true_positive)
	for(i=0; i<(int)target.size(); ++i)
		if(predictions[i] > 0)
		{
			n_pred_positive ++;
			if(target[i] > 0)
				n_true_positive ++;
		}

	double precision = (double)n_true_positive/n_pred_positive;
	return precision;
}

// Assumption: Number of elements in target and predictions are the same
double computeRecall(const vector<int>& target, const vector<int>& predictions)
{
	// Assert that the target and prediction vector have the same size
	assert(target.size() == predictions.size());

	int i;
	int n_target_positive = 0, n_true_positive = 0;

	for(i=0; i<(int)target.size(); ++i)
		if(target[i] > 0)
		{
			n_target_positive ++;
			if(predictions[i] > 0)
				n_true_positive ++;
		}

	double recall = (double)n_true_positive/n_target_positive;
	return recall;
}

double computeF1Score(const vector<int>& target, const vector<int>& predictions)
{
	// Assert that the target and prediction vector have the same size
	assert(target.size() == predictions.size());

	double precision, recall, f1_score;
	
	precision = computePrecision(target, predictions);
	recall = computeRecall(target, predictions);
	f1_score = 2*precision*recall/(precision + recall);

	return f1_score;
}