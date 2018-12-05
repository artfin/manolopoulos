#pragma once
#include <iostream>
#include <string>
#include <typeinfo>
#include <cxxabi.h>

//#define DEBUG_LOG_ENABLE_TIMING

#ifdef DEBUG_LOG_ENABLE_TIMING
#include <chrono>
#endif

#define DEBUG_PRINT(str) std::cout << str; 
#define DEBUG_METHOD(name) __log__ _logDebug(name);

class __log__
{
public:
    __log__(const std::string & context); 

    ~__log__();

    void fill_wsp();

    template<class T> void value_of(const std::string& name, const T& value );

    static inline std::string demangled_type_info_name(const std::type_info&ti)
    {
        int status = 0;
        return abi::__cxa_demangle(ti.name(),0,0,&status);
    }

private:
    static int wsp_len;
    const int wsp_step = 2;
    std::string wsp;
    std::string context;
#ifdef DEBUG_LOG_ENABLE_TIMING
    std::chrono::high_resolution_clock::time_point start_time;
#endif
};


