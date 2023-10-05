#include "PointCloud.hpp"
#include <fstream>
#include <iostream>

Eigen::MatrixXd ImportXYZFileToMatrix(const std::string &path_to_pc) {
  std::ifstream data(path_to_pc);
  if (data.is_open()) {
    // Read data from file
    std::vector<std::vector<std::string>> parsedData;
    std::string line;
    while (getline(data, line)) {
      std::stringstream lineStream(line);
      std::string cell; // single value
      std::vector<std::string> parsedRow;
      while (getline(lineStream, cell, ' ')) {
        parsedRow.push_back(cell);
      }
      parsedData.push_back(parsedRow);
    }

    // Check if each line contains exactly 3 values
    for (uint i = 0; i < parsedData.size(); i++) {
      if (parsedData[i].size() != 3) {
        std::cerr << "Line " << i + 1 << " does not contain exactly 3 values!"
                  << std::endl;
        exit(-1);
      }
    }

    // Create eigen array
    Eigen::MatrixXd X(parsedData.size(), 3);
    for (uint i = 0; i < parsedData.size(); i++) {
      for (uint j = 0; j < parsedData[i].size(); j++) {
        try {
          X(i, j) = stod(parsedData[i][j]);
        } catch (std::exception &e) {
          std::cerr << "Conversion of " << parsedData[i][j]
                    << " on row/column=" << i << "/" << j << " is not possible!"
                    << std::endl;
          exit(-1);
        }
      }
    }

    return X;
  } else {
    std::cerr << "Error opening file!" << std::endl;
    exit(-1);
  }
}

int main() {
  auto X_fix = ImportXYZFileToMatrix(
      "/home/yxu83/projects/redwood-mapping/data/dragon1.xyz");
  auto X_mov = ImportXYZFileToMatrix(
      "/home/yxu83/projects/redwood-mapping/data/dragon2.xyz");

  // Print first 5 rows
  std::cout << "X_fix:" << std::endl;
  std::cout << X_fix.block(0, 0, 5, 3) << std::endl;

  std::cout << "X_mov:" << std::endl;
  std::cout << X_mov.block(0, 0, 5, 3) << std::endl;

  printf("Create point cloud objects ...\n");
  PointCloud pc_fix{X_fix};
  PointCloud pc_mov{X_mov};

  return EXIT_SUCCESS;
}
