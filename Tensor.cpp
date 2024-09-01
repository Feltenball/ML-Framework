//
// Created by Maria on 8/27/2024.
//

#include "Tensor.h"

/* ---------- Tensor Creation Helper Routines ---------- */

FTensor zeros(std::vector<int> shape) {
    // Tensors cannot be jagged, so the size of the tensor is the product of shape values
    int size = 1;
    for (int dim : shape) {
        size = size * dim;
    }
    return FTensor(shape, std::vector<float>(size, 0));
}

FTensor from_vector(std::vector<float> vector) {
    return FTensor({static_cast<int>(vector.size()), 1},vector);
}

FTensor empty() {
    return FTensor({}, {});
}

/* ---------- Tensor Class Constructor ---------- */

FTensor::FTensor(std::vector<int> shape, std::vector<float> data)
    : stride_offset(0)
{
    // Get the amount of underlying data implied by the shape vector and check compatibility at time of tensor creation
    int n_data = 1;
    for (int dim : shape) { n_data = n_data * dim; }
    if (data.size() != n_data) { throw std::invalid_argument("data size mismatch"); }

    this->data = data;
    this->tensor_shape = shape;

    // The tensor strides can be obtained from the shape
    for (int stride_idx = 0; stride_idx < this->tensor_shape.size(); stride_idx++) {
        int stride = 1;

        for (int shape_idx = stride_idx + 1; shape_idx < this->tensor_shape.size(); shape_idx++) {
            stride = stride * this->tensor_shape[shape_idx];
        }
        this->tensor_strides.push_back(stride);
    }
}

/* ---------- Getter methods ---------- */

int FTensor::size() {
    return this->data.size();
}

std::vector<int> FTensor::shape() {
    return this->tensor_shape;
}

std::vector<float> FTensor::get_data() {
    return this->data;
}

/* ---------- Operator Overloading ---------- */

float& FTensor::operator () (std::vector<int> indexes) {
    int physical_idx = stride_offset;

    // Check the length of indexes is compatible with tensor_shape
    if (indexes.size() != this->tensor_shape.size()) { throw std::invalid_argument("indexes size mismatch shape size"); }
    // Check we are not out of range of any tensor dimensions
    // For loop will now not error because of above check
    for (int i = 0; i < this->tensor_shape.size(); i++) {
        if (indexes[i] >= this->tensor_shape[i]) { throw std::out_of_range("Tensor index out of range"); }
    }

    for (int i = 0; i < indexes.size(); i++) {
        physical_idx += indexes[i] * (tensor_strides[i]);
    }

    return data[physical_idx];
}

FTensor FTensor::operator + (FTensor &other) {
    // Two tensors must have exactly the same shape to admit addition
    if (this->tensor_shape != other.tensor_shape) { throw std::invalid_argument("Tensor shape mismatch"); }

    std::vector<float> result = std::vector<float>();

    for (int i = 0; i < other.size(); i++) {
        result.push_back(this->data[i] + other.data[i]);
    }

    return FTensor(this->tensor_shape, result);
}

FTensor FTensor::operator - (FTensor &other) {
    // Two tensors must have exactly the same shape to admit subtraction
    if (this->tensor_shape != other.tensor_shape) { throw std::invalid_argument("Tensor shape mismatch"); }

    std::vector<float> result = std::vector<float>();

    for (int i = 0; i < other.size(); i++) {
        result.push_back(this->data[i] - other.data[i]);
    }

    return FTensor(this->tensor_shape, result);
}

FTensor FTensor::operator * (FTensor &other) {
    std::vector<float> result = std::vector<float>();

    if (tensor_shape.size() == 2 && other.tensor_shape.size() == 2) {
        // Matrix multiplication

        int L_rows = this->tensor_shape[0];
        int L_cols = this->tensor_shape[1];
        int R_rows = other.tensor_shape[0];
        int R_cols = other.tensor_shape[1];

        if (L_cols != R_rows) { throw std::invalid_argument("2D tensors are not conformal for multiplication"); }

        std::vector<int> result_shape = {tensor_shape[0], other.tensor_shape[1]};
        FTensor result = zeros(result_shape);

        // Loop over entries of the result matrix and fill them in with the appropriate dot product
        for (int i = 0; i < result_shape[0]; i++) {
            for (int j = 0; j < result_shape[1]; j++) {
                float this_entry = 0;

                // Compute the dot product
                for (int k = 0; k < L_cols; k++) {
                    float L_elt = this->operator()({i, k});
                    float R_elt = other({k, j});
                    this_entry += L_elt * R_elt;
                }

                result({i, j}) = this_entry;

            }
        }

        return result;
    } else {
        throw std::invalid_argument("Tensor multiplication undefined for L shape and R shape");
    }
}

/* ---------- Tensor Mathematics / Linear Algebra ---------- */

FTensor FTensor::T() {
    // This _copy is necessary to avoid the shape of the original tensor being altered
    std::vector<int> tensor_shape_copy = tensor_shape;
    std::reverse(tensor_shape_copy.begin(), tensor_shape_copy.end());
    return FTensor(tensor_shape_copy, data);
}