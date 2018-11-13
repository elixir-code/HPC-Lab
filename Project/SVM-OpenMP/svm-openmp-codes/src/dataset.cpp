#include "dataset.hpp"

#include <fstream>
#include <string>

/* Read a classification dataset from a libsvm file format */
template <typename T>
Dataset<T> readLibsvmDataset(const char *filename, int n_data, int n_features)
{
	Dataset<T> dataset;

	ifstream file(filename);
	if(file.is_open())
	{
		string line, feature;
		size_t current_pos, delimiter_pos, feature_delimiter_pos;

		int i = 0, target, feature_index;
		T feature_value;
		vector<T> feature_vector(n_features);

		while((i < n_data || n_data == -1) && getline(file, line)) 
		{	
			fill(feature_vector.begin(), feature_vector.end(), 0);

			delimiter_pos = line.find(' ');
			target = stoi(line.substr(0, delimiter_pos));

			current_pos = delimiter_pos + 1;

			// skip the spaces after target
			while((current_pos < line.length()) && (line[current_pos] == ' '))
				current_pos++;

			while((delimiter_pos != string::npos) && (current_pos < line.length()))
			{
				delimiter_pos = line.find(' ', current_pos);
				feature = line.substr(current_pos, delimiter_pos - current_pos);

				feature_delimiter_pos = feature.find(':');

				feature_index = stoi(feature.substr(0, feature_delimiter_pos));
				feature_value = (T)stod(feature.substr(feature_delimiter_pos + 1));
				feature_vector[feature_index-1] = feature_value;

				current_pos = delimiter_pos + 1;
			}

			dataset.data.push_back(feature_vector);
			dataset.target.push_back(target);

			i++;
		}

		dataset.n_data = i;
		dataset.n_features = n_features;
	}

	return dataset;
}

// Explicit declaration of templated class and function
template class Dataset<int>;
template Dataset<int> readLibsvmDataset<int>(const char *, int, int);

template class Dataset<float>;
template Dataset<float> readLibsvmDataset<float>(const char *, int, int);

template class Dataset<double>;
template Dataset<double> readLibsvmDataset<double>(const char *, int, int);