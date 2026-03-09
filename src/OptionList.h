#pragma once

#include <map>
#include <string>
#include <vector>

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
public:
    void addString(const std::string & label, const std::string & desc, const std::string & value );
    void addInt   (const std::string & label, const std::string & desc, const int &         value );
    void addReal  (const std::string & label, const std::string & desc, const double &      value );
    void addSwitch(const std::string & label, const std::string & desc, const bool &        value );

    bool isString(const std::string & label) const;
    bool isInt(const std::string &    label) const;
    bool isReal(const std::string &   label) const;
    bool isSwitch(const std::string & label) const;

    std::string getString(const std::string & label) const;
    const int& getInt    (const std::string & label) const;
    double getReal       (const std::string & label) const;
    bool getSwitch       (const std::string & label) const;

    void setString(const std::string & label, const std::string & value);
    void setInt   (const std::string & label, const int &         value);
    void setReal  (const std::string & label, const double &      value);
    void setSwitch(const std::string & label, const bool &        value);

    std::string getInfo(const std::string & label) const;

    OptionList& operator=(const OptionList & other)
    {
        if (this != &other)        {
            m_strings = other.m_strings;
            m_ints    = other.m_ints;
            m_reals   = other.m_reals;
            m_switches = other.m_switches;
        }
        return *this;
    }

    std::vector<double> realValues() const;
};