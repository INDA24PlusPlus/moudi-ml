#pragma once

#include "network.h"
#include <gsl/gsl_vector_double.h>

size_t evaluate(Network * network, gsl_vector input);
void train(Network * network, gsl_vector input, size_t expected_result_index);
