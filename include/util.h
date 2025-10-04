#ifndef __util__
#define __util__
#include <fstream>
#include <iostream>
#include <string>
#include "json.hpp"
using json = nlohmann::json;

void save_json(std::string file_name, const json &j)
{
    std::ofstream output_file(file_name);
    if (output_file.is_open())
    {
        output_file << j.dump(4);
        output_file.close();
        std::cout << "Successfully saved JSON to " << file_name << std::endl;
    }
    else
    {
        std::cerr << "Error: Unable to open file " << file_name << " for writing." << std::endl;
    }
}

json load_json(const std::string &file_name)
{
    std::ifstream input_file(file_name);
    if (!input_file.is_open())
    {
        std::cerr << "Error: Could not open file " << file_name << std::endl;
        return nullptr;
    }
    try
    {
        json data = json::parse(input_file);
        std::cout << "Successfully parsed JSON from " << file_name << std::endl;
        return data;
    }
    catch (const json::parse_error &e)
    {
        std::cerr << "Error: JSON parsing failed in " << file_name << ": " << e.what() << std::endl;
        return nullptr;
    }
}

#endif