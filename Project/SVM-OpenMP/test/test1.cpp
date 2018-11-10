#include <iostream>
using namespace std;

#include "dataset.hpp"
#include "classifier.hpp"
#include "kernels.hpp"
#include "validation.hpp"

#include <omp.h>

int main(int argc, char *argv[])
{
	// Read training dataset from file
	Dataset<int> training_dataset = readLibsvmDataset<int>("./datasets/a1a", -1, 123);

	if(training_dataset.data.size() == 0)
	{
		cout << "ERROR: Failed to read training dataset from file" << endl;
		return 1;
	}

	// Create the SVM Classifier with parameters ana fit the data
	double C, tol, eps;
	C = 1.0;
	tol = 0.001;
	eps = 0.001;

	SVC<int> classifier(C, tol, eps, dotProduct);

	double start_time, end_time;

	start_time = omp_get_wtime();
	classifier.fit(training_dataset);
	end_time = omp_get_wtime();

	cout << "Fitted SVM model with training data in " << (end_time - start_time) << " seconds." << endl;

	// Read the testing dataset from file
	Dataset<int> testing_dataset = readLibsvmDataset<int>("./datasets/a1a.t", -1, 123);

	if(testing_dataset.data.size() == 0)
	{
		cout << "ERROR: Failed to read testing dataset from file" << endl;
		return 1;
	}

	start_time = omp_get_wtime();
	vector<int> predictions = classifier.predict(testing_dataset);
	end_time = omp_get_wtime();

	cout << "Predicted class labels for testing data in " << (end_time - start_time) << " seconds." << endl;

	// Compute the accuracy, precision, recall and f1-score of the classifier
	double accuracy, precision, recall, f1_score;
	accuracy = computeAccuracy(testing_dataset.target, predictions);
	precision = computePrecision(testing_dataset.target, predictions);
	recall = computeRecall(testing_dataset.target, predictions);
	f1_score = computeF1Score(testing_dataset.target, predictions);

	cout << "Accuracy = " << accuracy << endl;
	cout << "Precision = " << precision << endl;
	cout << "Recall = " << recall << endl;
	cout << "F1-Score = " << f1_score << endl;

	return 0;
}