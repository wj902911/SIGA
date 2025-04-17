#pragma once

#include <map>
#include <string>

class OptionList
{
private:
    typedef std::pair<std::string,std::string> StringOpt;
    typedef std::pair<int        ,std::string> IntOpt;
    typedef std::pair<double     ,std::string> RealOpt;
    typedef std::pair<bool       ,std::string> SwitchOpt;

    typedef std::map<std::string,StringOpt> StringTable;
    typedef std::map<std::string,   IntOpt> IntTable;
    typedef std::map<std::string,  RealOpt> RealTable;
    typedef std::map<std::string,SwitchOpt> SwitchTable;

    StringTable m_strings; 
    IntTable    m_ints;     
    RealTable   m_reals;    
    SwitchTable m_switches; 
};