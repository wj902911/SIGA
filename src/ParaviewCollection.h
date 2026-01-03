#pragma once
#include <fstream>
#include <sstream>
#include <cassert>

class ParaviewCollection
{
public:
    ParaviewCollection(std::string const& fn)
        : m_fn(fn), m_counter(0)
    {
        m_fn.append(".pvd");
    }

    void initalize()
    {
    	m_file << "<?xml version=\"1.0\"?>\n";
    	m_file << "<VTKFile type=\"Collection\" version=\"0.1\">";
    	m_file << "<Collection>\n";
    }

    void readFile()
    {
    	std::ifstream f(m_fn.c_str());
    	std::string line;
    	while (getline(f, line))
    	{
    		if (line == "</Collection>")
    			break;
    		m_file << line << "\n";
    	}
    	f.close();
    	std::ofstream of("test.txt");
    	of << m_file.str();
    	of.close();
    }

    void addTimestep(std::string const& fn, int tstep, std::string const& ext)
    {
    	m_file << "<DataSet timestep=\"" << tstep << "\" file=\"" << fn << ext << "\"/>\n";
    }

    void addTimestep(std::string const& fn, int part, int tstep, std::string const& ext)
    {
    	m_file << "<DataSet part=\""
    		<< part << "\" timestep=\""
    		<< tstep << "\" file=\""
    		<< fn << "_" << part << ext << "\"/>\n";
    }

    void saveStep()
    {
    	std::ofstream f(m_fn.c_str());
    	f << m_file.str();
    	f << "</Collection>\n";
    	f << "</VTKFile>\n";
    	f.close();
    }

	void save()
	{
		assert(m_counter != -1);
		m_file << "</Collection>\n";
		m_file << "</VTKFile>\n";

		std::ofstream f(m_fn.c_str());
		assert(f.is_open());
		f << m_file.str();
		f.close();
		m_file.str("");
		m_counter = -1;
	}

private:
    std::stringstream m_file;
    std::string m_fn;
    int m_counter;
};