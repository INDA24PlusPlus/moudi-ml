#pragma once

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

typedef struct layer {
    gsl_matrix weights;
    gsl_vector biases;
    gsl_vector activations;

    gsl_vector * bias_gradient;
    gsl_matrix * weights_gradient;
} Layer;

typedef struct network {
    double learning_rate;
    size_t batch_size;

    gsl_vector inputs;

    struct hidden_layer {
        Layer * layers;
        size_t count;
    } hidden;

    Layer output;
} Network;

#define LEARNING_RATE 0.1
/* #define SIGMOID_APPROX */

#ifdef SIGMOID_APPROX
#define SIGMOID(x) (0.5 * (1.0 + (x) / (1.0 + fabs(x))))
#define D_SIGMOID(x) (1.0 / (2.0 * powf(1.0 + fabs(x), 2.0)))
#else
#define SIGMOID(x) (1.0 / (1.0 + exp(-x)))
#define D_SIGMOID(z) (z * (1.0 - z))
#endif


#define _RELU_CONSTANT 0.01
#define RELU(x) (((x) < 0.0) ? _RELU_CONSTANT * x : x)
#define D_RELU(x) (((x) < 0.0) ? _RELU_CONSTANT : 1.0)

#define VECTOR_APPLY(dest, src, f) \
    for (size_t _vec_el_index = 0; _vec_el_index < (dest).size; ++_vec_el_index) {\
        double _propogate_layer_value = gsl_vector_get(&(src), _vec_el_index); \
        gsl_vector_set(&(dest), _vec_el_index, f(_propogate_layer_value)); \
    }

Network network_init(size_t inputs, size_t hidden_layers, size_t outputs, size_t batch_size);
Layer network_init_layer(size_t next_layer_nodes,  size_t nodes);

void network_feedforward(Network * network, gsl_vector values);
void network_back_prop(Network * network, size_t correct_output_index);
void network_update(Network * network, size_t batch_size);
