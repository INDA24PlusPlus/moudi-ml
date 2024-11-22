#include "network.h"
#include "fmt.h"

#include <math.h>
#include <gsl/gsl_blas.h>
#include <stdlib.h>
#include <time.h>

Network network_init(size_t inputs, size_t hidden_layers, size_t outputs, size_t batch_size) {
    if (hidden_layers == 0) {
        println("Network must have more than 0 hidden layers.");
        exit(1);
    }

    Network network = {
        .learning_rate = LEARNING_RATE,
        .batch_size = batch_size,
        .hidden = (struct hidden_layer) {
            .layers = calloc(sizeof(*(network.hidden.layers)), hidden_layers),
            .count = hidden_layers,
        },
    };

    size_t hidden_size = inputs * 2 / 3 + outputs;

    // input to hidden
    network.hidden.layers[0] = network_init_layer(inputs, hidden_size);

    // hidden to hidden
    for (size_t i = 1; i < hidden_layers; ++i) {
        network.hidden.layers[i] = network_init_layer(hidden_size, hidden_size);
    }

    // hidden to output
    network.output = network_init_layer(hidden_size, outputs);

    return network;
}

#define RAND_BOUND(upper) ((double)(rand() % upper) / (double)upper)

Layer network_init_layer(size_t incoming, size_t nodes) {
    Layer layer = {
        .weights = *gsl_matrix_alloc(nodes, incoming),
        .activations = *gsl_vector_calloc(nodes),
        .biases = *gsl_vector_calloc(nodes),
        .bias_gradient = NULL,
        .weights_gradient = NULL,
    };

    double u_bound = 1 / sqrt(incoming);
    double l_bound = -u_bound;

    srand(time(NULL));

    for (size_t i = 0; i < nodes; ++i) {
        for (size_t j = 0; j < incoming; ++j) {
            double value = ((u_bound - l_bound) * RAND_BOUND(RAND_MAX)) + l_bound;
            gsl_matrix_set(&layer.weights, i, j, value);
        }
    }

    return layer;
}

// 532x784 * 784*1 + 532x1

#define PROPOGATE_LAYER(layer, src, activation_func) \
    gsl_blas_dgemv(CblasNoTrans, 1.0, &((layer)->weights), &(src), 0.0, &((layer)->activations)); \
    gsl_blas_daxpy(1.0, &((layer)->biases), &(layer)->activations); \
    VECTOR_APPLY((layer)->activations, (layer)->activations, activation_func); \

void network_feedforward(Network * network, gsl_vector values) {
    Layer * prev_layer, * layer = &network->hidden.layers[0];
    network->inputs = values;

    // calculate first hidden layer
    PROPOGATE_LAYER(layer, network->inputs, SIGMOID);

    // propogate through hidden layer
    for (size_t i = 1; i < network->hidden.count; ++i) {
        prev_layer = layer;
        layer = &network->hidden.layers[i];
        PROPOGATE_LAYER(layer, prev_layer->activations, SIGMOID);
    }

    // calculate output layer
    PROPOGATE_LAYER(&network->output, layer->activations, SIGMOID);
}

void network_back_prop(Network * network, size_t correct_output_index) {
    Layer * layer, * prev_layer = &network->output;

    gsl_vector * output_error = gsl_vector_alloc(prev_layer->activations.size);

    const size_t activations = prev_layer->activations.size;
    for (size_t i = 0; i < activations; ++i) {
        double predicted = gsl_vector_get(&prev_layer->activations, i);
        double actual = (i == correct_output_index) ? 1.0 : 0.0;
        double error = (predicted - actual) * D_SIGMOID(predicted);
        gsl_vector_set(output_error, i, error);
    }

    gsl_vector * current_error = output_error;

    // since it will overflow we check if the size if less than the size still
    for (ssize_t i = network->hidden.count; i >= 0; --i) {
        layer = prev_layer;

        if (layer->bias_gradient == NULL || layer->weights_gradient == NULL) {
            if (layer->bias_gradient != NULL || layer->weights_gradient != NULL) {
                println("PANIC: Bias and Weights gradient are misaligned");
                exit(1);
            }

            layer->bias_gradient = gsl_vector_calloc(layer->biases.size);
            layer->weights_gradient = gsl_matrix_calloc(layer->weights.size1, layer->weights.size2);
        }

        // m = incoming, n = outgoing
        // dE/dW = dE/dY * X^T
        //  mxn     mx1  * 1xn 
        if (i > 0) {
            prev_layer = &network->hidden.layers[i - 1];
            if (gsl_blas_dger(1.0, current_error, &prev_layer->activations, layer->weights_gradient)) {
                println("dger");
                exit(1);
            }
        } else {
            prev_layer = NULL;
            if (gsl_blas_dger(1.0, current_error, &network->inputs, layer->weights_gradient)) {
                println("dger");
                exit(1);
            }
        }

        gsl_blas_daxpy(1.0, current_error, layer->bias_gradient);

        if (i > 0) {
            // E_{i-1} = W_i^T * E_i
            // m = incoming, n = outgoing
            // dE/dX = W^T * dE/dY
            //  nx1    nxm * mx1
            gsl_vector * prev_error = gsl_vector_alloc(prev_layer->activations.size);
            if (gsl_blas_dgemv(CblasTrans, 1.0, &layer->weights, current_error, 0.0, prev_error)) {
                println("dgemv");
                exit(1);
            }

            // E_{i-1} *= dRelu(z_{i-1})
            for (size_t j = 0; j < prev_error->size; ++j) {
                double z = gsl_vector_get(&prev_layer->activations, j);
                double error = gsl_vector_get(prev_error, j) * D_SIGMOID(z);
                /* println("z = {d}, err = {d}", z, error); */
                gsl_vector_set(prev_error, j, error);
            }

            gsl_vector_free(current_error);
            current_error = prev_error;
        }
    }
}

void network_update(Network * network, size_t batch_size) {
    for (size_t i = 0; i <= network->hidden.count; ++i) {
        Layer * layer = (i == network->hidden.count) ? &network->output : network->hidden.layers + i;
        if (layer->weights_gradient == NULL) {
            println("Layer weights gradient is invalidly NULL");
            exit(1);
        }

        if (layer->bias_gradient == NULL) {
            println("Layer bias gradient is invalidly NULL");
            exit(1);
        }

        gsl_matrix_scale(layer->weights_gradient, -network->learning_rate / (double) batch_size);
        gsl_vector_scale(layer->bias_gradient, -network->learning_rate / (double) batch_size);

        gsl_matrix_add(&layer->weights, layer->weights_gradient);
        gsl_vector_add(&layer->biases, layer->bias_gradient);

        gsl_matrix_free(layer->weights_gradient);
        gsl_vector_free(layer->bias_gradient);

        layer->weights_gradient = NULL;
        layer->bias_gradient = NULL;

    }

}
