#ifndef __util__
#define __util__
#include <fstream>
#include <iostream>
#include <string>
#include "json.hpp"
using json = nlohmann::json;

void save_json(std::string file_name, const json &j);

json load_json(const std::string &file_name);

double square(double x);

const double PI = 3.14159265358979323846;

#endif