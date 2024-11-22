#include "stddef.h"
#include "stdlib.h"

#include "gsl/gsl_vector.h"

struct mnisc_image {
    char pixels[28 * 28];
};

struct mnisc_label {
    char label;
};

struct mnisc_set {
    size_t size;
    struct mnisc_image * images;
    struct mnisc_label * labels;
};

struct mnisc_set new_mnisc_set(size_t size);
void parse_image_file(const char * path, struct mnisc_set * set);
void parse_label_file(const char * path, struct mnisc_set * set);
gsl_vector * mnisc_image_to_gsl_vector(struct mnisc_image image);
