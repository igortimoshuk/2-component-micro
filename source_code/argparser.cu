#include "argparser.h"


//
// Loop through argv and look for inputs matching "flag=(integer)".
//  Ex:  --n-sites=15  loops=1543  -cores=32
//
// NOTE: input is parsed by regex so avoid using special characters!
//
int input::parse_arg_i(int argc, char* argv[], const std::string flag, int default_value){
  std::regex reg_pattern(flag+"=(-?\\d+)");
  for (int index = 1 ; index < argc ; index++){
    std::string line (argv[index]);
    std::smatch match;
    std::regex_match(line, match, reg_pattern);
    if (match.size() > 0){
      return std::stoi(match.str(1));
    }
  }
  return default_value;
}



//
// Loop through argv and look for inputs matching "flag=(double)".
//  Ex:  --T=0.234  a=5.43  -H=1.4e-15
//
// NOTE: input is parsed by regex so avoid using special characters!
//
double input::parse_arg_d(int argc, char* argv[], const std::string flag, double default_value){
  std::regex reg_pattern(flag+"=(-?\\d+(\\.\\d+)?([eE][-+]?\\d+)?)");
  for (int index = 1 ; index < argc ; index++){
    std::string line (argv[index]);
    std::smatch match;
    std::regex_match(line, match, reg_pattern);
    if (match.size() > 0){
      return std::stod(match.str(1));
    }
  }
  return default_value;
}


//
// Loop through argv and look for inputs matching "flag=(string)".
//  Ex:  --outfile=data.dat  -path=datafolder  infile=input
//
// NOTE: input is parsed by regex so avoid using special characters!
//
std::string input::parse_arg_s(int argc, char* argv[], const std::string flag, std::string default_value){
  std::regex reg_pattern(flag+"=(.+)");
  for (int index = 1 ; index < argc ; index++){
    std::string line (argv[index]);
    std::smatch match;
    std::regex_match(line, match, reg_pattern);
    if (match.size() > 0){
      return match.str(1);
    }
  }
  return default_value;
}


//
// Loop through argv and look for inputs matching "flag".
//  Ex:  "--periodic"  "-no-visual"  "check"
//
// NOTE: input is parsed by regex so avoid using special characters!
//
bool input::parse_arg_bool(int argc, char* argv[], const std::string flag, bool default_value){
  std::regex reg_pattern(flag);
  for (int index = 1 ; index < argc ; index++){
    std::string line (argv[index]);
    std::smatch match;
    std::regex_match(line, match, reg_pattern);
    if (match.size() > 0){
      return !default_value;
    }
  }
  return default_value;
}
