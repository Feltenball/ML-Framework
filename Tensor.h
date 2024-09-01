//
// Created by Maria on 8/27/2024.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <algorithm>
#include <iostream>


class FTensor {
private:
    std::vector<float> data;
    std::vector<int> tensor_shape;
    std::vector<int> tensor_strides;
    int stride_offset;

public:
    FTensor(std::vector<int> shape, std::vector<float> data);
    ~FTensor() = default;

    /* ---------- Getter Methods ---------- */

    int size();
    std::vector<int> shape();
    std::vector<float> get_data();

    /* ---------- Operator Overloading ---------- */

    // Accessor
    float& operator () (std::vector<int> _shape);

    // Arithmetic
    FTensor operator + (FTensor &other);
    FTensor operator - (FTensor &other);
    FTensor operator * (FTensor &other);

    /* ---------- Tensor Mathematics / Linear Algebra ---------- */

    FTensor T();
};

/* ---------- Tensor Creation Helper Routines ---------- */

FTensor empty();
FTensor zeros(std::vector<int> shape);
FTensor from_matrix(std::vector<std::vector<float>> matrix);
FTensor from_vector(std::vector<float> vector);

/* ---------- Tensor Printing ---------- */

inline std::ostream& operator << (std::ostream &out, FTensor& tensor) {
    // TODO: This will error if the tensor's shape is empty

    for (int i = 0; i < tensor.shape()[0]; i++) {
        for (int j = 0; j < tensor.shape()[1]; j++) {
            out << tensor({i, j}) << " ";
        }
        out << std::endl;
    }
    return out;
}

#endif //TENSOR_H
