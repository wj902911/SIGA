#include <TeeLogger.h>

#include <cstdio>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace
{
#ifdef _WIN32
constexpr int stdoutFd = 1;
constexpr int stderrFd = 2;
#else
constexpr int stdoutFd = STDOUT_FILENO;
constexpr int stderrFd = STDERR_FILENO;
#endif
}

TeeLogger::TeeLogger(const std::string& logPath)
    : m_log(logPath, std::ios::out | std::ios::trunc)
{
    if (!m_log)
        throw std::runtime_error("Failed to open log file: " + logPath);

    start();
}

TeeLogger::~TeeLogger()
{
    stop();
}

std::ostream& TeeLogger::stream()
{
    return std::cout;
}

TeeLogger& TeeLogger::operator<<(std::ostream& (*manipulator)(std::ostream&))
{
    stream() << manipulator;
    return *this;
}

TeeLogger& TeeLogger::operator<<(std::ios_base& (*manipulator)(std::ios_base&))
{
    stream() << manipulator;
    return *this;
}

int TeeLogger::duplicateFd(int fd)
{
#ifdef _WIN32
    return _dup(fd);
#else
    return dup(fd);
#endif
}

int TeeLogger::duplicateToFd(int source, int target)
{
#ifdef _WIN32
    return _dup2(source, target);
#else
    return dup2(source, target);
#endif
}

int TeeLogger::closeFd(int fd)
{
#ifdef _WIN32
    return _close(fd);
#else
    return close(fd);
#endif
}

std::ptrdiff_t TeeLogger::readFd(int fd, char* buffer, std::size_t size)
{
#ifdef _WIN32
    return _read(fd, buffer, static_cast<unsigned int>(size));
#else
    return read(fd, buffer, size);
#endif
}

std::ptrdiff_t TeeLogger::writeFd(int fd, const char* buffer, std::size_t size)
{
#ifdef _WIN32
    return _write(fd, buffer, static_cast<unsigned int>(size));
#else
    return write(fd, buffer, size);
#endif
}

int TeeLogger::createPipe(int pipeFds[2])
{
#ifdef _WIN32
    return _pipe(pipeFds, 4096, _O_BINARY);
#else
    return pipe(pipeFds);
#endif
}

void TeeLogger::start()
{
    std::cout.flush();
    std::cerr.flush();
    std::fflush(stdout);
    std::fflush(stderr);

    int pipeFds[2] = {-1, -1};
    if (createPipe(pipeFds) != 0)
        throw std::runtime_error("Failed to create logging pipe.");

    m_savedStdout = duplicateFd(stdoutFd);
    m_savedStderr = duplicateFd(stderrFd);
    if (m_savedStdout < 0 || m_savedStderr < 0)
    {
        closeFd(pipeFds[0]);
        closeFd(pipeFds[1]);
        throw std::runtime_error("Failed to duplicate console output handles.");
    }

    if (duplicateToFd(pipeFds[1], stdoutFd) != 0 ||
        duplicateToFd(pipeFds[1], stderrFd) != 0)
    {
        closeFd(pipeFds[0]);
        closeFd(pipeFds[1]);
        closeFd(m_savedStdout);
        closeFd(m_savedStderr);
        throw std::runtime_error("Failed to redirect console output to log tee.");
    }

    m_pipeRead = pipeFds[0];
    closeFd(pipeFds[1]);
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    m_active = true;
    m_reader = std::thread(&TeeLogger::readLoop, this);
}

void TeeLogger::stop()
{
    if (!m_active)
        return;

    std::cout.flush();
    std::cerr.flush();
    std::fflush(stdout);
    std::fflush(stderr);

    duplicateToFd(m_savedStdout, stdoutFd);
    duplicateToFd(m_savedStderr, stderrFd);

    if (m_reader.joinable())
        m_reader.join();

    closeFd(m_savedStdout);
    closeFd(m_savedStderr);
    closeFd(m_pipeRead);

    m_log.flush();
    m_active = false;
}

void TeeLogger::readLoop()
{
    char buffer[4096];
    while (true)
    {
        const auto count = readFd(m_pipeRead, buffer, sizeof(buffer));
        if (count <= 0)
            break;

        writeAll(m_savedStdout, buffer, static_cast<std::size_t>(count));
        {
            std::lock_guard<std::mutex> lock(m_logMutex);
            m_log.write(buffer, count);
            m_log.flush();
        }
    }
}

void TeeLogger::writeAll(int fd, const char* buffer, std::size_t size)
{
    std::size_t written = 0;
    while (written < size)
    {
        const auto count = writeFd(fd, buffer + written, size - written);
        if (count <= 0)
            break;
        written += static_cast<std::size_t>(count);
    }
}
