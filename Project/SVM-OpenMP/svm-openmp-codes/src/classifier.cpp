#include "classifier.hpp"

template <typename T>
vector<int> Classifier<T>::predict(const Dataset<T>& dataset)
{
	int prediction;
	vector<int> predictions;

	for(int i=0; i<(int)dataset.data.size(); ++i)
	{
		prediction = predict(dataset.data[i]);
		predictions.push_back(prediction);
	}

	return predictions;
}

// Explicit instantiation of templated class and function
template class Classifier<int>;
template class Classifier<float>;
template class Classifier<double>;