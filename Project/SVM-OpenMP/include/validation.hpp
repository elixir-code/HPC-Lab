#ifndef VALIDATION_HPP
#define VALIDATION_HPP

#include <vector>
using namespace std;

// Computes the accuracy of the predictions by the classifier against true target values
double computeAccuracy(const vector<int>& target, const vector<int>& predictions);

// Computes the precision of the predictions by the classifier against true target values
double computePrecision(const vector<int>& target, const vector<int>& predictions);

// Computes the recall of the predictions by the classifier against true target values
double computeRecall(const vector<int>& target, const vector<int>& predictions);

// Computes the F1-score of the predictions by the classifier against true target values
double computeF1Score(const vector<int>& target, const vector<int>& predictions);

#endif