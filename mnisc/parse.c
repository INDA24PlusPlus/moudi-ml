#include "stddef.h"
#include "fmt.h"
#include <endian.h>
#include <gsl/gsl_vector_double.h>
#include <stdio.h>

#include "include/parse.h"

struct mnisc_set new_mnisc_set(size_t size) {
    return (struct mnisc_set) {
        .size = size,
        .images = calloc(sizeof(struct mnisc_image), size),
        .labels = calloc(sizeof(struct mnisc_label), size),
    };
}

#define READ_BE(dest, fp) \
    fread(&(dest), sizeof(dest), 1, fp);\
    dest = htobe32(dest)

void parse_image_file(const char * path, struct mnisc_set * set) {
	FILE * fp;
    char * buffer;
    size_t size;

    fp = fopen(path, "rb");

    if (fp == NULL) {
        println("[Error]: Cannot open file: '{s}'", path);
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    rewind(fp);
    
    unsigned int magic_number = 0;
    unsigned int image_count = 0;
    unsigned int number_of_rows = 0;
    unsigned int number_of_columns = 0;

    READ_BE(magic_number, fp);
    
    if (magic_number != 0x00000803) {
        println("Invalid image file provided: '{s}'", path);
        println("Magic number: {x}, {x}", magic_number, 0x00000803);
        exit(1);
    }

    READ_BE(image_count, fp);

    if (image_count != set->size) {
        println("Set and source file have misaligned sizes. Expected {i}, got {i}", set->size, image_count);
        exit(1);
    }

    READ_BE(number_of_rows, fp);

    if (number_of_rows != 28) {
        println("Invalid image size: rows = {i}", number_of_rows);
        exit(1);
    }

    READ_BE(number_of_columns, fp);

    if (number_of_columns != 28) {
        println("Invalid image size: columns = {i}", number_of_columns);
        exit(1);
    }

    fread(set->images, sizeof(struct mnisc_image), image_count, fp);
	fclose(fp);
}

void parse_label_file(const char * path, struct mnisc_set * set) {
	FILE * fp;
    char * buffer;
    size_t size;

    fp = fopen(path, "rb");

    if (fp == NULL) {
        println("[Error]: Cannot open file: '{s}'", path);
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    rewind(fp);

    unsigned int magic_number = 0;
    unsigned int label_count = 0;

    fread(&magic_number, sizeof(magic_number), 1, fp);
    magic_number = htobe32(magic_number); // change to big endian
    
    if (magic_number != 0x00000801) {
        println("Invalid label file provided: '{s}'", path);
        exit(1);
    }

    fread(&label_count, sizeof(label_count), 1, fp);
    label_count = htobe32(label_count); // change to big endian

    if (label_count != set->size) {
        println("Set and source file have misaligned sizes. Expected {i}, got {i}", set->size, label_count);
        exit(1);
    }

    fread(set->labels, sizeof(struct mnisc_image), label_count, fp);

	fclose(fp);
}

gsl_vector * mnisc_image_to_gsl_vector(struct mnisc_image image) {
    const size_t size = sizeof(image.pixels) / (sizeof(image.pixels[0]));
    gsl_vector * vector = gsl_vector_alloc(size);

    for (size_t i = 0; i < size; ++i) {
        gsl_vector_set(vector, i, (image.pixels[i] & 0xff) / 255.0);
    }

    return vector;
}
