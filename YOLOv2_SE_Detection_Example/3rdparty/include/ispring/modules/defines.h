#if defined(_WIN32)
#define ISPRING_WINDOWS

#if defined(_MSC_VER)
#define ISPRING_WINDOWS_VISUAL_STUDIO
#elif defined(__GNUC__)
#define ISPRING_WINDOWS_GCC
#endif
#define SELECT_ANY	__declspec(selectany)
#elif defined(__linux__)
#define ISPRING_LINUX
#if defined(__GNUC__)
#define ISPRING_LINUX_GCC
#endif
#define SELECT_ANY	__attribute__((weak))
#endif
