#include "log.hpp"

int __log__::wsp_len = 0;

__log__::__log__(const std::string & context) : context(context)
#ifdef DEBUG_LOG_ENABLE_TIMING
, start_time(std::chrono::high_resolution_clock::now())
#endif
{
    fill_wsp();
    std::cout << wsp << "--> " << context;
#ifdef DEBUG_LOG_ENABLE_TIMING
    auto end_time = std::chrono::high_resolution_clock::now(); 
    std::cout << " took " << 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0 << " s"; 
#endif
    std::cout << "\n";

    wsp_len += wsp_step;
} 

__log__::~__log__()
{
    wsp_len -= wsp_step;
    fill_wsp();
    std::cout << wsp << "<-- " << context << std::endl;
}

void __log__::fill_wsp()
{
    wsp = "";
    for ( int i = 0; i < wsp_len; ++i )
        wsp += " ";
}
