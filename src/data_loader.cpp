#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>

using Sample = std::vector<float>;
using Dataset = std::vector<std::pair<Sample, float>>;

Dataset load_dataset(const std::string& filename, int window_size) {
    std::ifstream file(filename);
    Dataset dataset;
    std::vector<float> series;

    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << "\n";
        return dataset;
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        ++line_num;
        std::stringstream ss(line);
        std::string time_str, value_str;
        if (std::getline(ss, time_str, ',') && std::getline(ss, value_str)) {
            try {
                float value = std::stof(value_str);
                if (std::isnan(value) || std::isinf(value)) throw std::runtime_error("Invalid number");
                series.push_back(value);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Skipping invalid line " << line_num << " in " << filename
                          << " (" << e.what() << ")\n";
            }
        }
    }

    for (size_t i = 0; i + window_size < series.size(); ++i) {
        Sample input(series.begin() + i, series.begin() + i + window_size);
        float output = series[i + window_size];
        dataset.emplace_back(input, output);
    }

    std::cout << "Loaded " << dataset.size() << " samples from " << filename << "\n";
    return dataset;
}
