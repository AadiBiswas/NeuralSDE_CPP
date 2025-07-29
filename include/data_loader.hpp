#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <vector>
#include <string>

using Sample = std::vector<float>;
using Dataset = std::vector<std::pair<Sample, float>>;

Dataset load_dataset(const std::string& filename, int window_size);

#endif
