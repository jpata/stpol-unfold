#ifndef __LOGGING_H__
#define __LOGGING_H__

#include <cstdlib>
#include <string>
#include <algorithm>
#include <cctype>

enum LOGLEVEL
{
    DEBUG=0,
    INFO=1,
    WARNING=2,
    ERROR=3,
    CRITICAL=4,
    NONE=5
};

std::string toStr(LOGLEVEL level)
{
    static std::string debug("DEBUG");
    static std::string info("INFO");
    static std::string warning("WARNING");
    static std::string error("ERROR");
    static std::string critical("CRITICAL");
    switch (level)
    {
        case DEBUG:
            return debug;
        case INFO:
            return info;
        case WARNING:
            return warning;
        case ERROR:
            return error;
        case CRITICAL:
            return critical;
        case NONE:
            return "";
    }
    return "";
    
}

static LOGLEVEL LogLevel = WARNING;

void setupLogging()
{
    const char* env = std::getenv("LOG_LEVEL");
    if (!env)
    {
        LogLevel=INFO;
        return;
    }
    std::string loglevel = std::string(env);
    std::transform(loglevel.begin(), loglevel.end(), loglevel.begin(),(int (*)(int))std::tolower);
    if (loglevel=="debug")
    {
        LogLevel=DEBUG;
    }
    else if (loglevel=="info")
    {
        LogLevel=INFO;
    }
    else if (loglevel=="warning")
    {
        LogLevel=DEBUG;
    }
    else if (loglevel=="error")
    {
        LogLevel=ERROR;
    }
    else if (loglevel=="critical")
    {
        LogLevel=CRITICAL;
    }
    else
    {
        LogLevel=NONE;
    }
}

template<class...ARGS> void log(LOGLEVEL level, const std::string& str, ARGS... args)
{
    if (level>=LogLevel)
    {
        if (str.substr(0,5) ==" ... ")
        {
            printf(str.c_str(),args...);
        }
        else
        {
            printf((toStr(level)+": "+str).c_str(),args...);
        }
    }
    fflush(stdout);
}
void log(LOGLEVEL level, const std::string& str)
{
    if (level>=LogLevel)
    {
        if (str.substr(0,4) ==" ... ")
        {
            printf("%s",str.c_str());
        }
        else
        {
            printf("%s",(toStr(level)+": "+str).c_str());
        }
    }
    fflush(stdout);
}

#endif

