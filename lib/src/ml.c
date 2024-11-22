#include "ml.h"

#include "network.h"
#include <gsl/gsl_blas.h>

size_t evaluate(Network * network, gsl_vector input) {
    network_feedforward(network, input);
    return gsl_blas_idamax(&network->output.activations);
}

void train(Network * network, gsl_vector input, size_t correct_output_index) {
    network_feedforward(network, input);
    network_back_prop(network, correct_output_index);
}
