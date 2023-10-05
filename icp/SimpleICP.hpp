#pragma once

#include <Eigen/Dense>

Eigen::MatrixXi KnnSearch(const Eigen::MatrixXd &X,
                          const Eigen::MatrixXd &X_query, const int &k = 1);
