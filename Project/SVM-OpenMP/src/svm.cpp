#include "classifier.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>

#include <iostream>
using namespace std;

/* SVM Classifier: Methods to initialize parameters, fit model and predict labels -- START */
template <typename T>
SVC<T>::SVC(double C, double tol, double eps, double (*kernelFunction)(vector<T>, vector<T>))
{
	SVC::C = C;
	SVC::tol = tol;
	SVC::kernelFunction = kernelFunction;
	SVC::eps = eps;

}

template <typename T>
void SVC<T>::fit(const Dataset<T>& dataset)
{
	// Copy the dataset to classifier
	SVC::dataset = dataset;

	int i, j;

	alphas = (double *)malloc(sizeof(double) * dataset.n_data);
	fill_n(alphas, dataset.n_data, 0);
	b = 0;

	// Calculate the error for each data sample
	errors = (double *)malloc(sizeof(double) * dataset.n_data);

	// #pragma omp parallel for default(none) private(i) shared(dataset, errors)
	#pragma omp parallel for private(i)
	for(i=0; i<dataset.n_data; ++i)
		errors[i] = -1 * dataset.target[i];

	// Perform optimization of alpha pairs using SMO (Sequential Minimal Optimization)
	int examine_all = 1, num_changed = 0;
	double kkt_parameter;

	while((num_changed > 0) || (examine_all == 1))
	{
		num_changed = 0;

		for(j=0; j<dataset.n_data; ++j)
		{
			// Iterates alternatively single scans through entire dataset and multiple scans through non-bound training examples
			if( (examine_all == 1) || ((alphas[j] > 0) && (alphas[j] < C)) )
			{
				// Identify the training examples (alpha j) that violate KKT conditions
				kkt_parameter = dataset.target[j] * errors[j];
				if( ((kkt_parameter < -1*tol) && (alphas[j] < C)) || ((kkt_parameter > tol) && (alphas[j] > 0)) )
					num_changed += findUpdateAlphaPair(j);
			}
		}

		if(examine_all == 1)
			examine_all = 0;

		else if(num_changed == 0)
			examine_all = 1;	
	}

	free(errors);
}

template <typename T>
int SVC<T>::predict(const vector<T>& data)
{	
	double output = 0;

	int i;

	// #pragma omp parallel for default(none) private(i) shared(dataset, alphas, kernelFunction, data) reduction(+:output)
	#pragma omp parallel for private(i) reduction(+:output)
	for(i=0; i<dataset.n_data; ++i)
		if(alphas[i] > 0)
			output += alphas[i] * dataset.target[i] * kernelFunction(dataset.data[i], data);

	output -= b;

	if(output >= 0)
		return 1;

	else
		return -1;
}

template <typename T>
int SVC<T>::findUpdateAlphaPair(int alpha2_index)
{
	int i;

	double estimated_step, max_estimated_step = -1;
	int alpha1_index = -1;

	// Iterate through all non-bound training examples to find other example for optimization using heuristic
	for(i=0; i<dataset.n_data; ++i)
		if((i!=alpha2_index) && (alphas[i] > 0) && (alphas[i] < C))
		{
			// Estimated step size of optimization
			estimated_step = errors[i] - errors[alpha2_index];
			if(estimated_step < 0)
				estimated_step *= -1;

			if(estimated_step > max_estimated_step)
				#pragma omp critical
				if(estimated_step > max_estimated_step)
				{
					max_estimated_step = estimated_step;
					alpha1_index = i;
				}
		}

	if((alpha1_index >= 0) && updateAlphaPair(alpha1_index, alpha2_index))
		return 1;

	// int random_offset = random() % dataset.n_data;

	// Parallely search across non-bound alphas for alpha1 that makes positive step
	valid_alpha1_found = 0;

	// #pragma omp parallel for default(none) private(i, alpha1_index) shared(dataset, alphas, alpha2_index, C)
	# pragma omp parallel
	{
		#pragma omp for private(i, alpha1_index)
		for(i=0; i<dataset.n_data; ++i)
		{	
			#pragma omp cancellation point for

			// alpha1_index = (i + random_offset) % dataset.n_data;
			alpha1_index = i;

			if((valid_alpha1_found == 0) && (alpha1_index != alpha2_index) && (alphas[alpha1_index] > 0) && (alphas[alpha1_index] < C))
			{
				// SVC<T>::updateAlphaPair has been expanded to circumvent oprphaned cancellation point problem
				#pragma omp cancellation point for

				if(valid_alpha1_found == 0)
				{
					double alpha1_value, alpha2_value, updated_alpha1_value, updated_alpha2_value, old_b_value;
					alpha1_value = alphas[alpha1_index];
					alpha2_value = alphas[alpha2_index];
					old_b_value = b;

					int s = dataset.target[alpha1_index] * dataset.target[alpha2_index];
					double L, H;

					if(s > 0)
					{
						L = max(0.0, alpha2_value + alpha1_value - C);
						H = min(C, alpha2_value + alpha1_value);
					}

					else
					{
						L = max(0.0, alpha2_value - alpha1_value);
						H = min(C, C + alpha2_value - alpha1_value);
					}

					if(L < H)
					{
						#pragma omp cancellation point for

						if(valid_alpha1_found == 0)
						{
							double k11, k12, k22, eta;

							k11 = kernelFunction(dataset.data[alpha1_index], dataset.data[alpha1_index]);
							k12 = kernelFunction(dataset.data[alpha1_index], dataset.data[alpha2_index]);
							k22 = kernelFunction(dataset.data[alpha2_index], dataset.data[alpha2_index]);

							eta = k11 + k22 - 2*k12;
							if(eta > 0)
							{
								updated_alpha2_value = alpha2_value + dataset.target[alpha2_index]*(errors[alpha1_index] - errors[alpha2_index])/eta;
								if(updated_alpha2_value < L) updated_alpha2_value = L;
								else if(updated_alpha2_value > H) updated_alpha2_value = H;
							}

							else
							{
								double f1, f2, L1, H1, Lobj, Hobj;
								f1 = dataset.target[alpha1_index]*(errors[alpha1_index]+b) - alpha1_value*k11 - s*alpha2_value*k12;
								f2 = dataset.target[alpha2_index]*(errors[alpha2_index]+b) - s*alpha1_value*k12 - alpha2_value*k22;

								L1 = alpha1_value + s*(alpha2_value - L);
								H1 = alpha1_value + s*(alpha2_value - H);

								Lobj = L1*f1 + L*f2 + (L1*L1*k11)/2 + (L*L*k22)/2 + s*L*L1*k12;
								Hobj = H1*f1 + H*f2 + (H1*H1*k11)/2 + (H*H*k22)/2 + s*H*H1*k12;

								if(Lobj < Hobj-eps)
									updated_alpha2_value = L;
								else if(Lobj > Hobj+eps)
									updated_alpha2_value = H;
								else
									updated_alpha2_value = alpha2_value;
							}

							if(fabs(updated_alpha2_value - alpha2_value) >= eps*(updated_alpha2_value + alpha2_value + eps))
							{
								int thread_valid_alpha1_found = -1;

								#pragma omp critical
								{
									if(valid_alpha1_found == 0)
									{
										valid_alpha1_found = 1;
										thread_valid_alpha1_found = 1;

										updated_alpha1_value = alpha1_value + s*(alpha2_value - updated_alpha2_value);

										// update the threshold value b
										if((updated_alpha1_value > 0) && (updated_alpha1_value < C))
											b += errors[alpha1_index] + dataset.target[alpha1_index]*(updated_alpha1_value - alpha1_value)*k11 + dataset.target[alpha2_index]*(updated_alpha2_value - alpha2_value)*k12;

										else if((updated_alpha2_value > 0) && (updated_alpha2_value < C))
											b += errors[alpha2_index] + dataset.target[alpha1_index]*(updated_alpha1_value - alpha1_value)*k12 + dataset.target[alpha2_index]*(updated_alpha2_value - alpha2_value)*k22;

										else
											b += (errors[alpha1_index] + errors[alpha2_index])/2 + dataset.target[alpha1_index]*(updated_alpha1_value - alpha1_value)*(k11+k12)/2 + dataset.target[alpha2_index]*(updated_alpha2_value - alpha2_value)*(k12+k22)/2;

										// update the error cache using new langrange multipliers
										double errors_delta_b, errors_delta_alpha1, errors_delta_alpha2;

										errors_delta_b = old_b_value - b;
										errors_delta_alpha1 = (updated_alpha1_value - alpha1_value) * dataset.target[alpha1_index];
										errors_delta_alpha2 = (updated_alpha2_value - alpha2_value) * dataset.target[alpha2_index];

										int i;
										
										// #pragma omp parallel for default(none) private(i) shared(dataset, errors, errors_delta_b, errors_delta_alpha1, errors_delta_alpha2, kernelFunction)
										#pragma omp parallel for private(i)
										for(i=0; i<dataset.n_data; ++i)
										{
											errors[i] += errors_delta_b;
											errors[i] += errors_delta_alpha1 * kernelFunction(dataset.data[alpha1_index], dataset.data[i]);
											errors[i] += errors_delta_alpha2 * kernelFunction(dataset.data[alpha2_index], dataset.data[i]);
										}

										// store alpha values in alphas array
										alphas[alpha1_index] = updated_alpha1_value;
										alphas[alpha2_index] = updated_alpha2_value;
									} // end of inner most if(valid_alpha1_found == 0)

								} // end of #pragma omp critical

								if(thread_valid_alpha1_found)
								{
									#pragma omp cancel for
								}
							}

						} // end of inner if(valid_alpha1_found == 0)
					} // end of if(L < H)
				} // end of outter if(valid_alpha1_found == 0)
			}
		}	
	}
	

	if(valid_alpha1_found)
		return 1;

	// Parallely search across all alphas for alpha1 that makes positive step
	valid_alpha1_found = 0;

	// #pragma omp parallel for default(none) private(i, alpha1_index) shared(dataset, alphas, alpha2_index, C)
	#pragma omp parallel
	{
		#pragma omp for private(i, alpha1_index)
		for(i=0; i<dataset.n_data; ++i)
		{	
			#pragma omp cancellation point for

			// alpha1_index = (i + random_offset) % dataset.n_data;
			alpha1_index = i;

			if((valid_alpha1_found == 0) && (alpha1_index != alpha2_index) && ((alphas[alpha1_index] == 0) || (alphas[alpha1_index] == C)) )
			{
				// SVC<T>::updateAlphaPair has been expanded to circumvent oprphaned cancellation point problem
				#pragma omp cancellation point for

				if(valid_alpha1_found == 0)
				{
					double alpha1_value, alpha2_value, updated_alpha1_value, updated_alpha2_value, old_b_value;
					alpha1_value = alphas[alpha1_index];
					alpha2_value = alphas[alpha2_index];
					old_b_value = b;

					int s = dataset.target[alpha1_index] * dataset.target[alpha2_index];
					double L, H;

					if(s > 0)
					{
						L = max(0.0, alpha2_value + alpha1_value - C);
						H = min(C, alpha2_value + alpha1_value);
					}

					else
					{
						L = max(0.0, alpha2_value - alpha1_value);
						H = min(C, C + alpha2_value - alpha1_value);
					}

					if(L < H)
					{
						#pragma omp cancellation point for

						if(valid_alpha1_found == 0)
						{
							double k11, k12, k22, eta;

							k11 = kernelFunction(dataset.data[alpha1_index], dataset.data[alpha1_index]);
							k12 = kernelFunction(dataset.data[alpha1_index], dataset.data[alpha2_index]);
							k22 = kernelFunction(dataset.data[alpha2_index], dataset.data[alpha2_index]);

							eta = k11 + k22 - 2*k12;
							if(eta > 0)
							{
								updated_alpha2_value = alpha2_value + dataset.target[alpha2_index]*(errors[alpha1_index] - errors[alpha2_index])/eta;
								if(updated_alpha2_value < L) updated_alpha2_value = L;
								else if(updated_alpha2_value > H) updated_alpha2_value = H;
							}

							else
							{
								double f1, f2, L1, H1, Lobj, Hobj;
								f1 = dataset.target[alpha1_index]*(errors[alpha1_index]+b) - alpha1_value*k11 - s*alpha2_value*k12;
								f2 = dataset.target[alpha2_index]*(errors[alpha2_index]+b) - s*alpha1_value*k12 - alpha2_value*k22;

								L1 = alpha1_value + s*(alpha2_value - L);
								H1 = alpha1_value + s*(alpha2_value - H);

								Lobj = L1*f1 + L*f2 + (L1*L1*k11)/2 + (L*L*k22)/2 + s*L*L1*k12;
								Hobj = H1*f1 + H*f2 + (H1*H1*k11)/2 + (H*H*k22)/2 + s*H*H1*k12;

								if(Lobj < Hobj-eps)
									updated_alpha2_value = L;
								else if(Lobj > Hobj+eps)
									updated_alpha2_value = H;
								else
									updated_alpha2_value = alpha2_value;
							}

							if(fabs(updated_alpha2_value - alpha2_value) >= eps*(updated_alpha2_value + alpha2_value + eps))
							{
								int thread_valid_alpha1_found = -1;

								#pragma omp critical
								{
									if(valid_alpha1_found == 0)
									{
										valid_alpha1_found = 1;
										thread_valid_alpha1_found = 1;

										updated_alpha1_value = alpha1_value + s*(alpha2_value - updated_alpha2_value);

										// update the threshold value b
										if((updated_alpha1_value > 0) && (updated_alpha1_value < C))
											b += errors[alpha1_index] + dataset.target[alpha1_index]*(updated_alpha1_value - alpha1_value)*k11 + dataset.target[alpha2_index]*(updated_alpha2_value - alpha2_value)*k12;

										else if((updated_alpha2_value > 0) && (updated_alpha2_value < C))
											b += errors[alpha2_index] + dataset.target[alpha1_index]*(updated_alpha1_value - alpha1_value)*k12 + dataset.target[alpha2_index]*(updated_alpha2_value - alpha2_value)*k22;

										else
											b += (errors[alpha1_index] + errors[alpha2_index])/2 + dataset.target[alpha1_index]*(updated_alpha1_value - alpha1_value)*(k11+k12)/2 + dataset.target[alpha2_index]*(updated_alpha2_value - alpha2_value)*(k12+k22)/2;

										// update the error cache using new langrange multipliers
										double errors_delta_b, errors_delta_alpha1, errors_delta_alpha2;

										errors_delta_b = old_b_value - b;
										errors_delta_alpha1 = (updated_alpha1_value - alpha1_value) * dataset.target[alpha1_index];
										errors_delta_alpha2 = (updated_alpha2_value - alpha2_value) * dataset.target[alpha2_index];

										int i;
										
										// #pragma omp parallel for default(none) private(i) shared(dataset, errors, errors_delta_b, errors_delta_alpha1, errors_delta_alpha2, kernelFunction)
										#pragma omp parallel for private(i)
										for(i=0; i<dataset.n_data; ++i)
										{
											errors[i] += errors_delta_b;
											errors[i] += errors_delta_alpha1 * kernelFunction(dataset.data[alpha1_index], dataset.data[i]);
											errors[i] += errors_delta_alpha2 * kernelFunction(dataset.data[alpha2_index], dataset.data[i]);
										}

										// store alpha values in alphas array
										alphas[alpha1_index] = updated_alpha1_value;
										alphas[alpha2_index] = updated_alpha2_value;

									} // end of inner most if(valid_alpha1_found == 0)

								} // end of #pragma omp critical

								if(thread_valid_alpha1_found)
								{
									#pragma omp cancel for
								}
							}

						} // end of inner if(valid_alpha1_found == 0)
					} // end of if(L < H)
				} // end of outter if(valid_alpha1_found == 0)
			}
		}
	} 
	

	if(valid_alpha1_found)
		return 1;

	return 0;
}

template <typename T>
int SVC<T>::updateAlphaPair(int alpha1_index, int alpha2_index)
{
	double alpha1_value, alpha2_value, updated_alpha1_value, updated_alpha2_value, old_b_value;
	alpha1_value = alphas[alpha1_index];
	alpha2_value = alphas[alpha2_index];
	old_b_value = b;

	int s = dataset.target[alpha1_index] * dataset.target[alpha2_index];
	double L, H;

	if(s > 0)
	{
		L = max(0.0, alpha2_value + alpha1_value - C);
		H = min(C, alpha2_value + alpha1_value);
	}

	else
	{
		L = max(0.0, alpha2_value - alpha1_value);
		H = min(C, C + alpha2_value - alpha1_value);
	}

	if(L == H)
		return 0;

	double k11, k12, k22, eta;

	k11 = kernelFunction(dataset.data[alpha1_index], dataset.data[alpha1_index]);
	k12 = kernelFunction(dataset.data[alpha1_index], dataset.data[alpha2_index]);
	k22 = kernelFunction(dataset.data[alpha2_index], dataset.data[alpha2_index]);

	eta = k11 + k22 - 2*k12;
	if(eta > 0)
	{
		updated_alpha2_value = alpha2_value + dataset.target[alpha2_index]*(errors[alpha1_index] - errors[alpha2_index])/eta;
		if(updated_alpha2_value < L) updated_alpha2_value = L;
		else if(updated_alpha2_value > H) updated_alpha2_value = H;
	}

	else
	{
		double f1, f2, L1, H1, Lobj, Hobj;
		f1 = dataset.target[alpha1_index]*(errors[alpha1_index]+b) - alpha1_value*k11 - s*alpha2_value*k12;
		f2 = dataset.target[alpha2_index]*(errors[alpha2_index]+b) - s*alpha1_value*k12 - alpha2_value*k22;

		L1 = alpha1_value + s*(alpha2_value - L);
		H1 = alpha1_value + s*(alpha2_value - H);

		Lobj = L1*f1 + L*f2 + (L1*L1*k11)/2 + (L*L*k22)/2 + s*L*L1*k12;
		Hobj = H1*f1 + H*f2 + (H1*H1*k11)/2 + (H*H*k22)/2 + s*H*H1*k12;

		if(Lobj < Hobj-eps)
			updated_alpha2_value = L;
		else if(Lobj > Hobj+eps)
			updated_alpha2_value = H;
		else
			updated_alpha2_value = alpha2_value;
	}

	if(fabs(updated_alpha2_value - alpha2_value) < eps*(updated_alpha2_value + alpha2_value + eps))
		return 0;

		
	updated_alpha1_value = alpha1_value + s*(alpha2_value - updated_alpha2_value);

	// update the threshold value b
	if((updated_alpha1_value > 0) && (updated_alpha1_value < C))
		b += errors[alpha1_index] + dataset.target[alpha1_index]*(updated_alpha1_value - alpha1_value)*k11 + dataset.target[alpha2_index]*(updated_alpha2_value - alpha2_value)*k12;

	else if((updated_alpha2_value > 0) && (updated_alpha2_value < C))
		b += errors[alpha2_index] + dataset.target[alpha1_index]*(updated_alpha1_value - alpha1_value)*k12 + dataset.target[alpha2_index]*(updated_alpha2_value - alpha2_value)*k22;

	else
		b += (errors[alpha1_index] + errors[alpha2_index])/2 + dataset.target[alpha1_index]*(updated_alpha1_value - alpha1_value)*(k11+k12)/2 + dataset.target[alpha2_index]*(updated_alpha2_value - alpha2_value)*(k12+k22)/2;

	// update the error cache using new langrange multipliers
	double errors_delta_b, errors_delta_alpha1, errors_delta_alpha2;

	errors_delta_b = old_b_value - b;
	errors_delta_alpha1 = (updated_alpha1_value - alpha1_value) * dataset.target[alpha1_index];
	errors_delta_alpha2 = (updated_alpha2_value - alpha2_value) * dataset.target[alpha2_index];

	int i;
		
	// #pragma omp parallel for default(none) private(i) shared(dataset, errors, errors_delta_b, errors_delta_alpha1, errors_delta_alpha2, kernelFunction)
	#pragma omp parallel for private(i)
	for(i=0; i<dataset.n_data; ++i)
	{
		errors[i] += errors_delta_b;
		errors[i] += errors_delta_alpha1 * kernelFunction(dataset.data[alpha1_index], dataset.data[i]);
		errors[i] += errors_delta_alpha2 * kernelFunction(dataset.data[alpha2_index], dataset.data[i]);
	}

	// store alpha values in alphas array
	alphas[alpha1_index] = updated_alpha1_value;
	alphas[alpha2_index] = updated_alpha2_value;
		
	return 1;
}
/* SVM Classifier: Methods to initialize parameters, fit model and predict labels -- END */

// explicit instantiation of template class and function
template class SVC<int>;
template class SVC<float>;
template class SVC<double>;