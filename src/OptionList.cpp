#include "OptionList.h"
#include <stdexcept>
#include <sstream>
#include "Utility_h.h"

#define ENSURE(cond, message) do if(!(cond)) {std::stringstream _m_;   \
    _m_<<"ENSURE `"<<#cond<<"` "<<message<<"\n"<<__FILE__<<", line "   \
     <<__LINE__<<" ("<< __FUNCTION__<< ")";                            \
    throw std::runtime_error(_m_.str());} while(false)

void OptionList::addString(const std::string &label, 
                           const std::string &desc, 
                           const std::string &value)
{
    ENSURE( !( isInt(label) || isReal(label) || isSwitch(label) ),
        "Invalid request (addString): Option "<<label<<" already exists, but not as a string; it is "<<getInfo(label)<<"." );
    m_strings[label] = std::make_pair(value,desc);
}

void OptionList::addInt(const std::string &label, 
                        const std::string &desc, 
                        const int &value)
{
    ENSURE( !( isString(label) || isReal(label) || isSwitch(label) ),
        "Invalid request (addInt): Option "<<label<<" already exists, but not as an int; it is "<<getInfo(label)<<"." );
    m_ints[label] = std::make_pair(value,desc);
}

void OptionList::addReal(const std::string &label, 
                         const std::string &desc, 
                         const double &value)
{
    ENSURE( !( isString(label) || isInt(label) || isSwitch(label) ),
         "Invalid request (addReal): Option "<<label<<" already exists, but not as a real; it is "<<getInfo(label)<<"." );
    m_reals[label] = std::make_pair(value,desc);
}

void OptionList::addSwitch(const std::string &label, 
                           const std::string &desc, 
                           const bool &value)
{
    ENSURE( !( isString(label) || isInt(label) || isReal(label) ),
         "Invalid request (addSwitch): Option "<<label<<" already exists, but not as a switch; it is "<<getInfo(label)<<"." );
    m_switches[label] = std::make_pair(value,desc);
}

std::string OptionList::getString(const std::string &label) const
{
    StringTable::const_iterator it = m_strings.find(label);
    ENSURE(it != m_strings.end(), "String option not found: " + label);
    return it->second.first;
}

const int & OptionList::getInt(const std::string & label) const
{
    IntTable::const_iterator it = m_ints.find(label);
    ENSURE(it!=m_ints.end(), "Invalid request (getInt): "+label+" is not an int; it is "+getString(label)+".");
    return it->second.first;
}

double OptionList::getReal(const std::string &label) const
{
    RealTable::const_iterator it = m_reals.find(label);
    ENSURE(it != m_reals.end(), "Invalid request (getReal): " + label + " is not a real; it is " + getString(label) + ".");
    return it->second.first;
}

bool OptionList::getSwitch(const std::string &label) const
{
    SwitchTable::const_iterator it = m_switches.find(label);
    ENSURE(it != m_switches.end(), "Invalid request (getSwitch): " + label + " is not a switch; it is " + getString(label) + ".");
    return it->second.first;
}

void OptionList::setString(const std::string &label, const std::string &value)
{
    StringTable::iterator it = m_strings.find(label);
    ENSURE(it!=m_strings.end(), "Invalid request (setString): "<<label<<" is not a string; it is "<<getInfo(label)<<".");
    it->second.first = value;
}

void OptionList::setInt(const std::string &label, const int &value)
{
    IntTable::iterator it = m_ints.find(label);
    ENSURE(it!=m_ints.end(), "Invalid request (setInt): " + label + " is not an int; it is " + getInfo(label) + ".");
    it->second.first = value;
}

void OptionList::setReal(const std::string &label, const double &value)
{
    RealTable::iterator it = m_reals.find(label);
    ENSURE(it!=m_reals.end(), "Invalid request (setReal): " + label + " is not a real; it is " + getInfo(label) + ".");
    it->second.first = value;
}

void OptionList::setSwitch(const std::string &label, const bool &value)
{
    SwitchTable::iterator it = m_switches.find(label);
    ENSURE(it!=m_switches.end(), "Invalid request (setSwitch): " + label + " is not a switch; it is " + getInfo(label) + ".");
    it->second.first = value;
}

std::string OptionList::getInfo(const std::string &label) const
{
    // find in strings
    StringTable::const_iterator it1 = m_strings.find(label);
    if ( it1 != m_strings.end() )
        return "a string (value:\"" + it1->second.first + "\")";

    // find in integers
    IntTable::const_iterator it2 = m_ints.find(label);
    if ( it2 != m_ints.end() )
        return "an int (value:" + to_string(it2->second.first) + ")";

    // find in reals
    RealTable::const_iterator it3 = m_reals.find(label);
    if ( it3 != m_reals.end() )
        return "a real (value:" + to_string(it3->second.first) + ")";

    // find in bools
    SwitchTable::const_iterator it4 = m_switches.find(label);
    if ( it4 != m_switches.end() )
        return "a switch (value:" + to_string(it4->second.first) + ")";

    return "undefined";
}

std::vector<double> OptionList::realValues() const
{
    std::vector<double> values;
    for (const auto& pair : m_reals)
        values.push_back(pair.second.first);
    return values;
}

bool OptionList::isString(const std::string &label) const
{ return m_strings.find(label) != m_strings.end(); }

bool OptionList::isInt(const std::string &label) const
{ return m_ints.find(label) != m_ints.end(); }

bool OptionList::isReal(const std::string &label) const
{ return m_reals.find(label) != m_reals.end(); }

bool OptionList::isSwitch(const std::string &label) const
{ return m_switches.find(label) != m_switches.end(); }
