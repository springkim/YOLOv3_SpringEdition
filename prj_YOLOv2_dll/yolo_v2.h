
#pragma comment(lib,"cublas.lib")
#pragma comment(lib,"curand.lib")
#pragma comment(lib,"cudart.lib")
#pragma comment(lib,"vfw32.lib")
#pragma comment( lib, "comctl32.lib" )
#pragma warning(disable:4099)
#pragma once

#ifndef CUDA_H
#define CUDA_H

#pragma warning(disable:4190)
#if defined(_MSC_VER) && _MSC_VER < 1900
#define inline __inline
#endif

extern int gpu_index;

#ifdef GPU

#define BLOCK 512

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#ifdef CUDNN
#include "cudnn.h"
#endif

void check_error(cudaError_t status);
cublasHandle_t blas_handle();
float *cuda_make_array(float *x, size_t n);
int *cuda_make_int_array(size_t n);
void cuda_push_array(float *x_gpu, float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
void cuda_set_device(int n);
void cuda_free(float *x_gpu);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif

#endif
#ifndef BOX_H
#define BOX_H

typedef struct {
	float x, y, w, h;
} box;

typedef struct {
	float dx, dy, dw, dh;
} dbox;

box float_to_box(float *f);
float box_iou(box a, box b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
#ifndef IMAGE_H
#define IMAGE_H

#include <cstdlib>
#include <cstdio>
#include <float.h>
#include <cstring>
#include <math.h>

typedef struct {
	int h;
	int w;
	int c;
	float *data;
} image;

float get_color(int c, int x, int max);
void flip_image(image a);
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
void draw_bbox(image a, box bbox, int w, float r, float g, float b);
void draw_label(image a, int r, int c, image label, const float *rgb);
void write_label(image a, int r, int c, image *characters, char *string, float *rgb);
void draw_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **labels, int classes);
image image_distance(image a, image b);
void scale_image(image m, float s);
image crop_image(image im, int dx, int dy, int w, int h);
image random_crop_image(image im, int w, int h);
image random_augment_image(image im, float angle, float aspect, int low, int high, int size);
void random_distort_image(image im, float hue, float saturation, float exposure);
image resize_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
void translate_image(image m, float s);
void normalize_image(image p);
image rotate_image(image m, float rad);
void rotate_image_cw(image im, int times);
void embed_image(image source, image dest, int dx, int dy);
void saturate_image(image im, float sat);
void exposure_image(image im, float sat);
void distort_image(image im, float hue, float sat, float val);
void saturate_exposure_image(image im, float sat, float exposure);
void hsv_to_rgb(image im);
void rgbgr_image(image im);
void constrain_image(image im);
void composite_3d(char *f1, char *f2, char *out, int delta);
int best_3d_shift_r(image a, image b, int min, int max);

image grayscale_image(image im);
image threshold_image(image im, float thresh);

image collapse_image_layers(image source, int border);
image collapse_images_horz(image *ims, int n);
image collapse_images_vert(image *ims, int n);

void show_image(image p, const char *name);
void show_image_normalized(image im, const char *name);
void save_image_png(image im, const char *name);
void save_image(image p, const char *name);
void show_images(image *ims, int n, char *window);
void show_image_layers(image p, char *name);
void show_image_collapsed(image p, char *name);

void print_image(image m);

image make_image(int w, int h, int c);
image make_random_image(int w, int h, int c);
image make_empty_image(int w, int h, int c);
image float_to_image(int w, int h, int c, float *data);
image copy_image(image p);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image **load_alphabet();

float get_pixel(image m, int x, int y, int c);
float get_pixel_extend(image m, int x, int y, int c);
void set_pixel(image m, int x, int y, int c, float val);
void add_pixel(image m, int x, int y, int c, float val);
float bilinear_interpolate(image im, float x, float y, int c);

image get_image_layer(image m, int l);

void free_image(image m);
void test_resize(char *filename);

#endif

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#pragma warning(disable:4244)

typedef enum {
	LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
}ACTIVATION;

ACTIVATION get_activation(char *s);

char *get_activation_string(ACTIVATION a);
float activate(float x, ACTIVATION a);
float gradient(float x, ACTIVATION a);
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);
void activate_array(float *x, const int n, const ACTIVATION a);
#ifdef GPU
void activate_array_ongpu(float *x, int n, ACTIVATION a);
void gradient_array_ongpu(float *x, int n, ACTIVATION a, float *delta);
#endif

static inline float stair_activate(float x)
{
	int n = floor(x);
	if (n % 2 == 0) return floor(x / 2.);
	else return (x - n) + floor(x / 2.);
}
static inline float hardtan_activate(float x)
{
	if (x < -1) return -1;
	if (x > 1) return 1;
	return x;
}
static inline float linear_activate(float x) { return x; }
static inline float logistic_activate(float x) { return 1. / (1. + exp(-x)); }
static inline float loggy_activate(float x) { return 2. / (1. + exp(-x)) - 1; }
static inline float relu_activate(float x) { return x*(x>0); }
static inline float elu_activate(float x) { return (x >= 0)*x + (x < 0)*(exp(x) - 1); }
static inline float relie_activate(float x) { return (x>0) ? x : .01*x; }
static inline float ramp_activate(float x) { return x*(x>0) + .1*x; }
static inline float leaky_activate(float x) { return (x>0) ? x : .1*x; }
static inline float tanh_activate(float x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }
static inline float plse_activate(float x)
{
	if (x < -4) return .01 * (x + 4);
	if (x > 4)  return .01 * (x - 4) + 1;
	return .125*x + .5;
}

static inline float lhtan_activate(float x)
{
	if (x < 0) return .001*x;
	if (x > 1) return .001*(x - 1) + 1;
	return x;
}
static inline float lhtan_gradient(float x)
{
	if (x > 0 && x < 1) return 1;
	return .001F;
}

static inline float hardtan_gradient(float x)
{
	if (x > -1 && x < 1) return 1;
	return 0;
}
static inline float linear_gradient(float x) { return 1; }
static inline float logistic_gradient(float x) { return (1 - x)*x; }
static inline float loggy_gradient(float x)
{
	float y = (x + 1.) / 2.;
	return 2 * (1 - y)*y;
}
static inline float stair_gradient(float x)
{
	if (floor(x) == x) return 0;
	return 1;
}
static inline float relu_gradient(float x) { return (x>0); }
static inline float elu_gradient(float x) { return (x >= 0) + (x < 0)*(x + 1); }
static inline float relie_gradient(float x) { return (x>0) ? 1 : .01; }
static inline float ramp_gradient(float x) { return (x>0) + .1; }
static inline float leaky_gradient(float x) { return (x>0) ? 1 : .1; }
static inline float tanh_gradient(float x) { return 1 - x*x; }
static inline float plse_gradient(float x) { return (x < 0 || x > 1) ? .01 : .125; }

#endif

#ifndef TREE_H
#define TREE_H

typedef struct {
	int *leaf;
	int n;
	int *parent;
	int *group;
	char **name;

	int groups;
	int *group_size;
	int *group_offset;
} tree;

tree *read_tree(char *filename);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves);
void change_leaves(tree *t, char *leaf_list);
float get_hierarchy_probability(float *x, tree *hier, int c);

#endif
#ifndef BASE_LAYER_H
#define BASE_LAYER_H


struct network_state;

struct layer;
typedef struct layer layer;

typedef enum {
	CONVOLUTIONAL,
	DECONVOLUTIONAL,
	CONNECTED,
	MAXPOOL,
	SOFTMAX,
	DETECTION,
	DROPOUT,
	CROP,
	ROUTE,
	COST,
	NORMALIZATION,
	AVGPOOL,
	LOCAL,
	SHORTCUT,
	ACTIVE,
	RNN,
	GRU,
	CRNN,
	BATCHNORM,
	NETWORK,
	XNOR,
	REGION,
	REORG,
	BLANK
} LAYER_TYPE;

typedef enum {
	SSE, MASKED, SMOOTH
} COST_TYPE;

struct layer {
	LAYER_TYPE type;
	ACTIVATION activation;
	COST_TYPE cost_type;
	void(*forward)   (struct layer, struct network_state);
	void(*backward)  (struct layer, struct network_state);
	void(*update)    (struct layer, int, float, float, float);
	void(*forward_gpu)   (struct layer, struct network_state);
	void(*backward_gpu)  (struct layer, struct network_state);
	void(*update_gpu)    (struct layer, int, float, float, float);
	int batch_normalize;
	int shortcut;
	int batch;
	int forced;
	int flipped;
	int inputs;
	int outputs;
	int truths;
	int h, w, c;
	int out_h, out_w, out_c;
	int n;
	int max_boxes;
	int groups;
	int size;
	int side;
	int stride;
	int reverse;
	int pad;
	int sqrt;
	int flip;
	int index;
	int binary;
	int xnor;
	int steps;
	int hidden;
	float dot;
	float angle;
	float jitter;
	float saturation;
	float exposure;
	float shift;
	float ratio;
	int softmax;
	int classes;
	int coords;
	int background;
	int rescore;
	int objectness;
	int does_cost;
	int joint;
	int noadjust;
	int reorg;
	int log;

	int adam;
	float B1;
	float B2;
	float eps;
	float *m_gpu;
	float *v_gpu;
	int t;
	float *m;
	float *v;

	tree *softmax_tree;
	int  *map;

	float alpha;
	float beta;
	float kappa;

	float coord_scale;
	float object_scale;
	float noobject_scale;
	float class_scale;
	int bias_match;
	int random;
	float thresh;
	int classfix;
	int absolute;

	int dontload;
	int dontloadscales;

	float temperature;
	float probability;
	float scale;

	int *indexes;
	float *rand;
	float *cost;
	char  *cweights;
	float *state;
	float *prev_state;
	float *forgot_state;
	float *forgot_delta;
	float *state_delta;

	float *concat;
	float *concat_delta;

	float *binary_weights;

	float *biases;
	float *bias_updates;

	float *scales;
	float *scale_updates;

	float *weights;
	float *weight_updates;

	float *col_image;
	int   * input_layers;
	int   * input_sizes;
	float * delta;
	float * output;
	float * squared;
	float * norms;

	float * spatial_mean;
	float * mean;
	float * variance;

	float * mean_delta;
	float * variance_delta;

	float * rolling_mean;
	float * rolling_variance;

	float * x;
	float * x_norm;

	struct layer *input_layer;
	struct layer *self_layer;
	struct layer *output_layer;

	struct layer *input_gate_layer;
	struct layer *state_gate_layer;
	struct layer *input_save_layer;
	struct layer *state_save_layer;
	struct layer *input_state_layer;
	struct layer *state_state_layer;

	struct layer *input_z_layer;
	struct layer *state_z_layer;

	struct layer *input_r_layer;
	struct layer *state_r_layer;

	struct layer *input_h_layer;
	struct layer *state_h_layer;

	float *z_cpu;
	float *r_cpu;
	float *h_cpu;

	float *binary_input;

	size_t workspace_size;

#ifdef GPU
	float *z_gpu;
	float *r_gpu;
	float *h_gpu;

	int *indexes_gpu;
	float * prev_state_gpu;
	float * forgot_state_gpu;
	float * forgot_delta_gpu;
	float * state_gpu;
	float * state_delta_gpu;
	float * gate_gpu;
	float * gate_delta_gpu;
	float * save_gpu;
	float * save_delta_gpu;
	float * concat_gpu;
	float * concat_delta_gpu;

	float *binary_input_gpu;
	float *binary_weights_gpu;

	float * mean_gpu;
	float * variance_gpu;

	float * rolling_mean_gpu;
	float * rolling_variance_gpu;

	float * variance_delta_gpu;
	float * mean_delta_gpu;

	float * col_image_gpu;

	float * x_gpu;
	float * x_norm_gpu;
	float * weights_gpu;
	float * weight_updates_gpu;

	float * biases_gpu;
	float * bias_updates_gpu;

	float * scales_gpu;
	float * scale_updates_gpu;

	float * output_gpu;
	float * delta_gpu;
	float * rand_gpu;
	float * squared_gpu;
	float * norms_gpu;
#ifdef CUDNN
	cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
	cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
	cudnnFilterDescriptor_t weightDesc;
	cudnnFilterDescriptor_t dweightDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionFwdAlgo_t fw_algo;
	cudnnConvolutionBwdDataAlgo_t bd_algo;
	cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

void free_layer(layer);

#endif
#ifndef MATRIX_H
#define MATRIX_H
typedef struct matrix {
	int rows, cols;
	float **vals;
} matrix;

matrix make_matrix(int rows, int cols);
void free_matrix(matrix m);
void print_matrix(matrix m);

matrix csv_to_matrix(char *filename);
void matrix_to_csv(matrix m);
matrix hold_out_matrix(matrix *m, int n);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
matrix resize_matrix(matrix m, int size);

float *pop_column(matrix *m, int c);

#endif
#ifndef LIST_H
#define LIST_H

typedef struct node {
	void *val;
	struct node *next;
	struct node *prev;
} node;

typedef struct list {
	int size;
	node *front;
	node *back;
} list;

list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);

void **list_to_array(list *l);

void free_list(list *l);
void free_list_contents(list *l);

#endif
#ifndef DATA_H
#define DATA_H
#include "pthread.h"

#if defined(_MSC_VER) && _MSC_VER < 1900
#define inline __inline
#endif

static inline float distance_from_edge(int x, int max)
{
	int dx = (max / 2) - x;
	if (dx < 0) dx = -dx;
	dx = (max / 2) + 1 - dx;
	dx *= 2;
	float dist = (float)dx / max;
	if (dist > 1) dist = 1;
	return dist;
}

typedef struct {
	int w, h;
	matrix X;
	matrix y;
	int shallow;
	int *num_boxes;
	box **boxes;
} data;

typedef enum {
	CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA
} data_type;

typedef struct load_args {
	int threads;
	char **paths;
	char *path;
	int n;
	int m;
	char **labels;
	int h;
	int w;
	int out_w;
	int out_h;
	int nh;
	int nw;
	int num_boxes;
	int min, max, size;
	int classes;
	int background;
	int scale;
	float jitter;
	float angle;
	float aspect;
	float saturation;
	float exposure;
	float hue;
	data *d;
	image *im;
	image *resized;
	data_type type;
	tree *hierarchy;
} load_args;

typedef struct {
	int id;
	float x, y, w, h;
	float left, right, top, bottom;
} box_label;

void free_data(data d);

pthread_t load_data(load_args args);

pthread_t load_data_in_thread(load_args args);

void print_letters(float *pred, int n);
data load_data_captcha(char **paths, int n, int m, int k, int w, int h);
data load_data_captcha_encode(char **paths, int n, int m, int w, int h);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure);
data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
data load_data_super(char **paths, int n, int m, int w, int h, int scale);
data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
data load_go(char *filename);

box_label *read_boxes(char *filename, int *n);
data load_cifar10_data(char *filename);
data load_all_cifar10();

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h);

list *get_paths(char *filename);
char **get_labels(char *filename);
void get_random_batch(data d, int n, float *X, float *y);
data get_data_part(data d, int part, int total);
data get_random_data(data d, int num);
void get_next_batch(data d, int n, int offset, float *X, float *y);
data load_categorical_data_csv(char *filename, int target, int k);
void normalize_data_rows(data d);
void scale_data_rows(data d, float s);
void translate_data_rows(data d, float s);
void randomize_data(data d);
data *split_data(data d, int part, int total);
data concat_data(data d1, data d2);
data concat_datas(data *d, int n);
void fill_truth(char *path, char **labels, int k, float *truth);

#endif
// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

typedef enum {
	CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network {
	float *workspace;
	int n;
	int batch;
	int *seen;
	float epoch;
	int subdivisions;
	float momentum;
	float decay;
	layer *layers;
	int outputs;
	float *output;
	learning_rate_policy policy;

	float learning_rate;
	float gamma;
	float scale;
	float power;
	int time_steps;
	int step;
	int max_batches;
	float *scales;
	int   *steps;
	int num_steps;
	int burn_in;

	int adam;
	float B1;
	float B2;
	float eps;

	int inputs;
	int h, w, c;
	int max_crop;
	int min_crop;
	float angle;
	float aspect;
	float exposure;
	float saturation;
	float hue;

	int gpu_index;
	tree *hierarchy;

#ifdef GPU
	float **input_gpu;
	float **truth_gpu;
#endif
} network;

typedef struct network_state {
	float *truth;
	float *input;
	float *delta;
	float *workspace;
	int train;
	int index;
	network net;
} network_state;

#ifdef GPU
float train_networks(network *nets, int n, data d, int interval);
void sync_nets(network *nets, int n, int interval);
float train_network_datum_gpu(network net, float *x, float *y);
float *network_predict_gpu(network net, float *input);
float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float *get_network_output_gpu(network net);
void forward_network_gpu(network net, network_state state);
void backward_network_gpu(network net, network_state state);
void update_network_gpu(network net);
#endif

float get_current_rate(network net);
int get_current_batch(network net);
void free_network(network net);
void compare_networks(network n1, network n2, data d);
char *get_layer_string(LAYER_TYPE a);

network make_network(int n);
void forward_network(network net, network_state state);
void backward_network(network net, network_state state);
void update_network(network net);

float train_network(network net, data d);
float train_network_batch(network net, data d, int n);
float train_network_sgd(network net, data d, int n);
float train_network_datum(network net, float *x, float *y);

matrix network_predict_data(network net, data test);
float *network_predict(network net, float *input);
float network_accuracy(network net, data d);
float *network_accuracies(network net, data d, int n);
float network_accuracy_multi(network net, data d, int n);
void top_predictions(network net, int n, int *index);
float *get_network_output(network net);
float *get_network_output_layer(network net, int i);
float *get_network_delta_layer(network net, int i);
float *get_network_delta(network net);
int get_network_output_size_layer(network net, int i);
int get_network_output_size(network net);
image get_network_image(network net);
image get_network_image_layer(network net, int i);
int get_predicted_class_network(network net);
void print_network(network net);
void visualize_network(network net);
int resize_network(network *net, int w, int h);
void set_batch_network(network *net, int b);
int get_network_input_size(network net);
float get_network_cost(network net);

int get_network_nuisance(network net);
int get_network_background(network net);

#endif
#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

typedef layer convolutional_layer;

#ifdef GPU
void forward_convolutional_layer_gpu(convolutional_layer layer, network_state state);
void backward_convolutional_layer_gpu(convolutional_layer layer, network_state state);
void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay);

void push_convolutional_layer(convolutional_layer layer);
void pull_convolutional_layer(convolutional_layer layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l);
#endif
#endif

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
void denormalize_convolutional_layer(convolutional_layer l);
void resize_convolutional_layer(convolutional_layer *layer, int w, int h);
void forward_convolutional_layer(const convolutional_layer layer, network_state state);
void update_convolutional_layer(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay);
image *visualize_convolutional_layer(convolutional_layer layer, char *window, image *prev_weights);
void binarize_weights(float *weights, int n, int size, float *binary);
void swap_binary(convolutional_layer *l);
void binarize_weights2(float *weights, int n, int size, char *binary, float *scales);

void backward_convolutional_layer(convolutional_layer layer, network_state state);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

image get_convolutional_image(convolutional_layer layer);
image get_convolutional_delta(convolutional_layer layer);
image get_convolutional_weight(convolutional_layer layer, int i);

int convolutional_out_height(convolutional_layer layer);
int convolutional_out_width(convolutional_layer layer);
void rescale_weights(convolutional_layer l, float scale, float trans);
void rgbgr_weights(convolutional_layer l);

#endif


#ifndef PARSER_H
#define PARSER_H

network parse_network_cfg(char *filename);
void save_network(network net, char *filename);
void save_weights(network net, char *filename);
void save_weights_upto(network net, char *filename, int cutoff);
void save_weights_double(network net, char *filename);
void load_weights(network *net, char *filename);
void load_weights_upto(network *net, char *filename, int cutoff);

#endif
#ifndef REGION_LAYER_H
#define REGION_LAYER_H

typedef layer region_layer;

region_layer make_region_layer(int batch, int h, int w, int n, int classes, int coords);
void forward_region_layer(const region_layer l, network_state state);
void backward_region_layer(const region_layer l, network_state state);
void get_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);
void resize_region_layer(layer *l, int w, int h);

#ifdef GPU
void forward_region_layer_gpu(const region_layer l, network_state state);
void backward_region_layer_gpu(region_layer l, network_state state);
#endif

#endif
#ifndef UTILS_H
#define UTILS_H
#include <cstdio>
#include <time.h>

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf(buf,len, format,...) _snprintf_s(buf, len,len, format, __VA_ARGS__)
#endif

#define SECRET_NUM -1234
#define TWO_PI 6.2831853071795864769252866

int *read_map(char *filename);
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
void free_ptrs(void **ptrs, int n);
char *basecfg(char *cfgfile);
int alphanum_to_int(char c);
char int_to_alphanum(int i);
int read_int(int fd);
void write_int(int fd, int n);
void read_all(int fd, char *buffer, size_t bytes);
void write_all(int fd, char *buffer, size_t bytes);
int read_all_fail(int fd, char *buffer, size_t bytes);
int write_all_fail(int fd, char *buffer, size_t bytes);
void find_replace(char *str, char *orig, char *rep, char *output);
void error(const char *s);
void malloc_error();
void file_error(char *s);
void strip(char *s);
void strip_char(char *s, char bad);
void top_k(float *a, int n, int k, int *index);
list *split_str(char *s, char delim);
char *fgetl(FILE *fp);
list *parse_csv_line(char *line);
char *copy_string(char *s);
int count_fields(char *line);
float *parse_fields(char *line, int n);
void normalize_array(float *a, int n);
void scale_array(float *a, int n, float s);
void translate_array(float *a, int n, float s);
int max_index(float *a, int n);
float constrain(float min, float max, float a);
int constrain_int(int a, int min, int max);
float mse_array(float *a, int n);
float rand_normal();
size_t rand_size_t();
float rand_uniform(float min, float max);
float rand_scale(float s);
int rand_int(int min, int max);
float sum_array(float *a, int n);
float mean_array(float *a, int n);
void mean_arrays(float **a, int n, int els, float *avg);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
float dist_array(float *a, float *b, int n, int sub);
float **one_hot_encode(float *a, int n, int k);
float sec(clock_t clocks);
int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
int sample_array(float *a, int n);
void print_statistics(float *a, int n);

#endif

#pragma once
#include<string>
#include<Windows.h>		//GetCurrentDirectoryA
#include<direct.h>		//_chdir
#include<vector>
#include<algorithm>
#include<opencv2/opencv.hpp>
extern void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear);
int YoloDetect(image img, int* _net, float threshold, float* result, int result_sz);
extern "C" {
	__declspec(dllexport) void YoloTrain(char* _base_dir, char* _datafile, char* _cfgfile);
	__declspec(dllexport) int* YoloLoad(char* cfgfile, char* weightsfile);
	__declspec(dllexport) int YoloDetectFromFile(char* img_path, int* _net, float threshold, float* result, int result_sz);
	//__declspec(dllexport) int YoloDetectFromCvMat(cv::Mat img, int* _net, float threshold, float* result, int result_sz);
	__declspec(dllexport) int YoloDetectFromBytesImage(unsigned char* img, int w, int h, int* _net, float threshold, float* result, int result_sz);

	__declspec(dllexport) void YoloTrainPy(wchar_t* _base_dir, wchar_t* _datafile, wchar_t* _cfgfile);
	__declspec(dllexport) int __stdcall YoloSizePy();
	__declspec(dllexport) void __stdcall YoloLoadPy(wchar_t* _cfgfile, wchar_t* _weightsfile,int* _network);
	__declspec(dllexport) int __stdcall YoloDetectFromFilePy(wchar_t* _img_path, int* _net, int _threshold, int result_sz);
	__declspec(dllexport) int  getClassPy(int index);
	__declspec(dllexport) int  getConfPy(int index);
	__declspec(dllexport) int  getXPy(int index);
	__declspec(dllexport) int  getYPy(int index);
	__declspec(dllexport) int  getWPy(int index);
	__declspec(dllexport) int  getHPy(int index);
}