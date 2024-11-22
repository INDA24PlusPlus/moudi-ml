#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>
#include <math.h>
#include <raylib.h>
#include <stdlib.h>
#include "fmt.h"

#include "include/parse.h"
#include "ml.h"
#include "network.h"

#define SCREEN_WIDTH 28 * 30
#define SCREEN_HEIGHT 28 * 20
#define TARGET_FPS 60

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_SCALE 5
#define IMAGE_SWITCH_TIMEOUT_SECONDS 1
#define IMAGE_SWITCH_TIMEOUT_FRAMES IMAGE_SWITCH_TIMEOUT_SECONDS * TARGET_FPS

void draw_mnisc_image(struct mnisc_image image) {
    int x = ((SCREEN_WIDTH - IMAGE_WIDTH * IMAGE_SCALE) / 2),
        y = ((SCREEN_HEIGHT - IMAGE_HEIGHT * IMAGE_SCALE) / 2);

    for (size_t y_offset = 0; y_offset < IMAGE_HEIGHT; ++y_offset) {
        for (size_t x_offset = 0; x_offset < IMAGE_WIDTH; ++x_offset) {
            char pixel = image.pixels[x_offset + IMAGE_WIDTH * y_offset];
            DrawRectangle(  x + x_offset * IMAGE_SCALE,
                            y + y_offset * IMAGE_SCALE,
                            IMAGE_SCALE,
                            IMAGE_SCALE,
                            (Color) {pixel, pixel, pixel, 255}
            );
        }
    }
}

double calculate_loss(gsl_vector predicted_vector, size_t correct) {
    double loss = 0.0;
    for (size_t i = 0; i < predicted_vector.size; ++i) {
        double predicted = gsl_vector_get(&predicted_vector, i);
        double actual = (i == correct) ? 1.0 : 0.0;
        double diff = (actual - predicted);
        loss += diff * diff;
    }

    return loss / predicted_vector.size;
}

void info(Network * network, struct mnisc_set * set, size_t index) {
    gsl_vector * input = mnisc_image_to_gsl_vector(set->images[index]);
    train(network, *input, set->labels[index].label);
    size_t result = evaluate(network, *input);
    gsl_vector_free(input);

    for (size_t i = 0; i < network->output.activations.size; ++i) {
        println("{i}: {d}", i, gsl_vector_get(&network->output.activations, i));
    }

    println("Image label: {i}, Predicted: {i}, Loss: {d}", set->labels[index], result, calculate_loss(network->output.activations, index));
}

void print_matrix(gsl_matrix matrix) {
    for (size_t i = 0; i < matrix.size1; ++i) {
        for (size_t j = 0; j < matrix.size2; ++j) {
            print("{d}  ", gsl_matrix_get(&matrix, i, j));
        }
        println("");
    }
}

void print_vector(gsl_vector vector, const char * str) {
    for (size_t i = 0; i < vector.size; ++i) {
        println("{s}[{i}]: {d}", str, i, gsl_vector_get(&vector, i));
    }
}

int main() {

    /* Network network = network_init(2, 1, 2, 1); */
    /*  */
    /* gsl_vector_set(&network.hidden.layers[0].biases, 0, 0.35); */
    /* gsl_vector_set(&network.hidden.layers[0].biases, 1, 0.35); */
    /*  */
    /* gsl_matrix_set(&network.hidden.layers[0].weights, 0, 0, 0.15); */
    /* gsl_matrix_set(&network.hidden.layers[0].weights, 0, 1, 0.20); */
    /* gsl_matrix_set(&network.hidden.layers[0].weights, 1, 0, 0.25); */
    /* gsl_matrix_set(&network.hidden.layers[0].weights, 1, 1, 0.30); */
    /*  */
    /* gsl_vector_set(&network.output.biases, 0, 0.60); */
    /* gsl_vector_set(&network.output.biases, 1, 0.60); */
    /*  */
    /* gsl_matrix_set(&network.output.weights, 0, 0, 0.40); */
    /* gsl_matrix_set(&network.output.weights, 0, 1, 0.45); */
    /* gsl_matrix_set(&network.output.weights, 1, 0, 0.50); */
    /* gsl_matrix_set(&network.output.weights, 1, 1, 0.55); */
    /*  */
    /* gsl_vector * input = gsl_vector_alloc(2); */
    /* gsl_vector_set(input, 0, 0.05); */
    /* gsl_vector_set(input, 1, 0.10); */
    /*  */
    /* evaluate(&network, *input); */
    /* train(&network, *input, 0); */
    /* network_update(&network, 1); */

    struct mnisc_set set = new_mnisc_set(60000);
    
    parse_image_file("./mnisc/data/train-images-idx3-ubyte", &set);
    parse_label_file("./mnisc/data/train-labels-idx1-ubyte", &set);
    
    struct mnisc_set test_set = new_mnisc_set(10000);
    
    parse_image_file("./mnisc/data/t10k-images-idx3-ubyte", &test_set);
    parse_label_file("./mnisc/data/t10k-labels-idx1-ubyte", &test_set);
    
    const size_t batch_size = 10;
    Network network = network_init(IMAGE_WIDTH * IMAGE_HEIGHT, 1, 10, batch_size);
    gsl_vector * inputs = malloc(sizeof(gsl_vector) * set.size);
    
    const size_t one_percent = set.size / 100;
    
    for (size_t cycle = 0; cycle < 1; ++cycle) {
        for (size_t i = 0; i < set.size; ++i) {
            if (cycle == 0) {
                inputs[i] = *mnisc_image_to_gsl_vector(set.images[i]);
            }

            train(&network, inputs[i], set.labels[i].label);
    
            if (i != 0) {
                if (i % batch_size) {
                    network_update(&network, batch_size);
                }
        
                if (i % one_percent == 0) {
                    println("Progress: {i}%, Loss: {d}", i / one_percent, calculate_loss(network.output.activations, set.labels[i].label));
                    /* info(&network, &set, i); */
                }
            }
        }
    }
    
    size_t correct = 0;
    
    for (size_t i = 0; i < test_set.size; ++i) {
        gsl_vector * input = mnisc_image_to_gsl_vector(test_set.images[i]);
        size_t result = evaluate(&network, *input);
    
        if (result == test_set.labels[i].label) {
            correct += 1;
        }
    } 

    println("Correct: {i}", correct);


    /* InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Machine Learning: MNISC"); */
    /* SetTargetFPS(TARGET_FPS); */
    /* info(&network, &set, image_index); */
    /*  */
    /* while (!WindowShouldClose()) { */
    /*     BeginDrawing(); */
    /*     ClearBackground(RAYWHITE); */
    /*  */
    /*     draw_mnisc_image(set.images[image_index]); */
    /*      */
    /*     if (time_since_last_switch >= IMAGE_SWITCH_TIMEOUT_FRAMES) { */
    /*         time_since_last_switch = 0; */
    /*         image_index += 1; */
    /*         info(&network, &set, image_index); */
    /*     } */
    /*  */
    /*     time_since_last_switch += 1; */
    /*  */
    /*     EndDrawing(); */
    /* } */
    /*  */
    /* CloseWindow(); */
}
