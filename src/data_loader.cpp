#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

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
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string time_str, value_str;
        if (std::getline(ss, time_str, ',') && std::getline(ss, value_str)) {
            series.push_back(std::stof(value_str));
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
