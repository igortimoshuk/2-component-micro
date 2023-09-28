#ifndef CUSTOM_ARG_PARSE
#define CUSTOM_ARG_PARSE

#include<string>
#include<regex>

namespace input{
//
// Loop through argv and look for inputs matching "flag=(integer)".
//  Ex:  --n-sites=15  loops=1543  -cores=32
//
// NOTE: input is parsed by regex so avoid using special characters!
//
extern int parse_arg_i(int argc, char* argv[], const std::string flag, int default_value);



//
// Loop through argv and look for inputs matching "flag=(double)".
//  Ex:  --T=0.234  a=5.43  -H=1.4e-15
//
// NOTE: input is parsed by regex so avoid using special characters!
//
extern double parse_arg_d(int argc, char* argv[], const std::string flag, double default_value);

//
// Loop through argv and look for inputs matching "flag=(string)".
//  Ex:  --outfile=data.dat  -path=datafolder  infile=input
//
// NOTE: input is parsed by regex so avoid using special characters!
//
extern std::string parse_arg_s(int argc, char* argv[], const std::string flag, std::string default_value);


//
// Loop through argv and look for inputs matching "flag".
//  Ex:  "--periodic"  "-no-visual"  "check"
//
// NOTE: input is parsed by regex so avoid using special characters!
//
extern bool parse_arg_bool(int argc, char* argv[], const std::string flag, bool default_value);

}


#endif