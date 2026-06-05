#pragma once

#include <cstddef>
#include <fstream>
#include <ios>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>

class TeeLogger
{
public:
    explicit TeeLogger(const std::string& logPath);
    ~TeeLogger();

    TeeLogger(const TeeLogger&) = delete;
    TeeLogger& operator=(const TeeLogger&) = delete;

    std::ostream& stream();

    template <typename T>
    TeeLogger& operator<<(const T& value)
    {
        stream() << value;
        return *this;
    }

    TeeLogger& operator<<(std::ostream& (*manipulator)(std::ostream&));
    TeeLogger& operator<<(std::ios_base& (*manipulator)(std::ios_base&));

private:
    static int duplicateFd(int fd);
    static int duplicateToFd(int source, int target);
    static int closeFd(int fd);
    static std::ptrdiff_t readFd(int fd, char* buffer, std::size_t size);
    static std::ptrdiff_t writeFd(int fd, const char* buffer, std::size_t size);
    static int createPipe(int pipeFds[2]);

    void start();
    void stop();
    void readLoop();
    void writeAll(int fd, const char* buffer, std::size_t size);

    std::ofstream m_log;
    std::mutex m_logMutex;
    std::thread m_reader;
    int m_savedStdout = -1;
    int m_savedStderr = -1;
    int m_pipeRead = -1;
    bool m_active = false;
};
