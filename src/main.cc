/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * Samuel Bourasseau wrote this file. You can do whatever you want with this
 * stuff. If we meet some day, and you think this stuff is worth it, you can
 * buy me a beer in return.
 * ----------------------------------------------------------------------------
 */

#include <array>
#include <iostream>
#include <memory>

#include <chrono>
#include <cstring>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <X11/extensions/Xfixes.h>
#include <X11/Xlib.h>
#include <X11/Xos.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>

#include "input.h"
#include "numtk.h"

using proc_glXCreateContextAttribsARB =
    GLXContext(*)(Display*, GLXFBConfig, GLXContext, Bool, int const*);
using proc_glXSwapIntervalEXT =
    void(*)(Display*, GLXDrawable, int);
using proc_glXSwapIntervalMESA =
    int(*)(unsigned);

static const int kVisualAttributes[] = {
    GLX_X_RENDERABLE, True,
    GLX_DOUBLEBUFFER, True,
    GLX_RENDER_TYPE, GLX_RGBA_BIT,
    GLX_SAMPLE_BUFFERS, 1,
    GLX_SAMPLES, 1,
    GLX_DEPTH_SIZE, 24,
    GLX_STENCIL_SIZE, 8,
#if 0
    GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
    GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
#endif
    None
};

static const int kGLContextAttributes[] = {
    GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
    GLX_CONTEXT_MINOR_VERSION_ARB, 5,
    //#define SR_GL_DEBUG_CONTEXT
#ifdef SR_GL_DEBUG_CONTEXT
    GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_DEBUG_BIT_ARB,
#endif
    None
};


static constexpr int boot_width = 1280;
static constexpr int boot_height = 720;

struct engine_module_t
{
    char const* path;
    void* hlib;
    time_t last_load_time;

    void (*run_frame_cb)(void*, void*);
    void (*init_cb)(void**);
    void (*shutdown_cb)(void*);
    void (*onreload_cb)(void*);
};
bool UpdateEngineModule(engine_module_t& _module);

int main(int __argc, char* __argv[])
{
    engine_module_t engine_main{};
    engine_main.path = "./libengine.so";

    Display * const display = XOpenDisplay(nullptr);
    if (!display)
    {
        std::cerr << "Can't connect to X server" << std::endl;
        return 1;
    }
    std::cout << "Connected to X server" << std::endl;


    int glx_major = -1, glx_minor = -1;
    if (!glXQueryVersion(display, &glx_major, &glx_minor))
    {
        std::cerr << "glXQueryVersion failed." << std::endl;
        return 1;
    }
    if (!(glx_major > 1 || (glx_major == 1 && glx_minor >= 3)))
    {
        std::cerr << "GLX 1.3 is required." << std::endl;
        return 1;
    }
    std::cout << "GLX " << glx_major << "." << glx_minor << " was found." << std::endl;


    GLXFBConfig selected_config{};
    {
        int fb_count = 0;
        GLXFBConfig* const fb_configs = glXChooseFBConfig(display, DefaultScreen(display),
                                                          kVisualAttributes, &fb_count);
        if (!fb_configs)
        {
            std::cerr << "No framebuffer configuration found." << std::endl;
            return 1;
        }
        std::cout << fb_count << " configs found" << std::endl;
        selected_config = fb_configs[0];
        XFree(fb_configs);
    }

    XVisualInfo* const visual_info = glXGetVisualFromFBConfig(display, selected_config);

    XSetWindowAttributes swa;
    swa.colormap = XCreateColormap(display,
                                   RootWindow(display, visual_info->screen),
                                   visual_info->visual, AllocNone);
    swa.border_pixel = 0;
    swa.event_mask = StructureNotifyMask;
    int const swa_mask = CWColormap | CWBorderPixel | CWEventMask;

    Window window = XCreateWindow(display, RootWindow(display, visual_info->screen),
                                   0, 0, boot_width, boot_height, 0, visual_info->depth, InputOutput,
                                   visual_info->visual,
                                   swa_mask, &swa);
    if (!window)
    {
        std::cerr << "Window creation failed" << std::endl;
        return 1;
    }
    XFree(visual_info);

    Atom const wm_delete_window = [](Display* display, Window window)
    {
        int wm_protocols_size = 0;
        Atom* wm_protocols = nullptr;
        Status result = XGetWMProtocols(display, window, &wm_protocols, &wm_protocols_size);
        std::cout << "XGetWMProtocols status " << std::to_string(result) << std::endl;
        std::cout << "protocols found " << std::to_string(wm_protocols_size) << std::endl;
        XFree(wm_protocols);

        Atom wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", True);
        if (wm_delete_window == None)
            std::cout << "[ERROR] WM_DELETE_WINDOW doesn't exist" << std::endl;
        result = XSetWMProtocols(display, window, &wm_delete_window, 1);
        std::cout << "XSetWMProtocols status " << std::to_string(result) << std::endl;

        result = XGetWMProtocols(display, window, &wm_protocols, &wm_protocols_size);
        std::cout << "XGetWMProtocols status " << std::to_string(result) << std::endl;
        std::cout << "protocols found " << std::to_string(wm_protocols_size) << std::endl;
        XFree(wm_protocols);

        int property_count = 0;
        Atom* x_properties = XListProperties(display, window, &property_count);
        std::cout << "properties found " << std::to_string(property_count) << std::endl;
        XFree(x_properties);

        return wm_delete_window;
    }(display, window);

    static constexpr long kEventMask =
        StructureNotifyMask
        | ButtonPressMask | ButtonReleaseMask
        | PointerMotionMask
        | KeyPressMask | KeyReleaseMask;
    XSelectInput(display, window, kEventMask);

    XStoreName(display, window, "x11_bootstrap");
    XMapWindow(display, window);

    auto const glXCreateContextAttribsARB = reinterpret_cast<proc_glXCreateContextAttribsARB>(
        glXGetProcAddressARB((GLubyte const*)"glXCreateContextAttribsARB")
        );
    if (!glXCreateContextAttribsARB)
    {
        std::cerr << "glXCreateContextAttribsARB procedure unavailable" << std::endl;
        return 1;
    }

    auto const glXSwapIntervalEXT = reinterpret_cast<proc_glXSwapIntervalEXT>(
        glXGetProcAddressARB((GLubyte const*)"glXSwapIntervalEXT")
        );
    if (!glXSwapIntervalEXT)
    {
        std::cerr << "glXSwapIntervalEXT procedure unavailable" << std::endl;
        return 1;
    }

    auto const glXSwapIntervalMESA = reinterpret_cast<proc_glXSwapIntervalMESA>(
        glXGetProcAddressARB((GLubyte const*)"glXSwapIntervalMESA")
        );

    GLXContext glx_context = glXCreateContextAttribsARB(display, selected_config, 0,
                                                     True, kGLContextAttributes);
    XSync(display, False);
    if (!glx_context)
    {
        std::cerr << "Modern GL context creation failed" << std::endl;
        return 1;
    }

    if (!glXIsDirect(display, glx_context))
    {
        std::cout << "Indirect GLX rendering context created" << std::endl;
    }

    glXMakeCurrent(display, window, glx_context);
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "glew failed to initalize." << std::endl;
    }

    glXSwapIntervalMESA(1);
    //glXSwapIntervalEXT(display, window, 1);
	std::cout << "GL init complete : " << std::endl;
	std::cout << "OpenGL version : " << glGetString(GL_VERSION) << std::endl;
	std::cout << "Manufacturer : " << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Drivers : " << glGetString(GL_RENDERER) << std::endl;
	{
		std::cout << "Context flags : ";
		int context_flags = 0;
		glGetIntegerv(GL_CONTEXT_FLAGS, &context_flags);
		if (context_flags & GL_CONTEXT_FLAG_DEBUG_BIT)
			std::cout << "debug, ";
		std::cout << std::endl;
	}

#ifdef SR_GL_DEBUG_CONTEXT
    oglbase::DebugMessageControl<> debugMessageControl{};
#endif

    glXMakeCurrent(display, 0, 0);

    // ===============================================================
    // ENGINE INIT
    // ===============================================================
    glXMakeCurrent(display, window, glx_context);

    UpdateEngineModule(engine_main);

    void* engine = nullptr;
    engine_main.init_cb(&engine);
    engine_main.onreload_cb(engine);

    std::array<eKey, 256> key_map{};
    key_map[(std::uint8_t)(XK_Tab & 0xff)] = eKey::kTab;
    key_map[(std::uint8_t)(XK_Left & 0xff)] = eKey::kLeft;
    key_map[(std::uint8_t)(XK_Right & 0xff)] = eKey::kRight;
    key_map[(std::uint8_t)(XK_Up & 0xff)] = eKey::kUp;
    key_map[(std::uint8_t)(XK_Down & 0xff)] = eKey::kDown;
    key_map[(std::uint8_t)(XK_Page_Up & 0xff)] = eKey::kPageUp;
    key_map[(std::uint8_t)(XK_Page_Down & 0xff)] = eKey::kPageDown;
    key_map[(std::uint8_t)(XK_Home & 0xff)] = eKey::kHome;
    key_map[(std::uint8_t)(XK_End & 0xff)] = eKey::kEnd;
    key_map[(std::uint8_t)(XK_Insert & 0xff)] = eKey::kInsert;
    key_map[(std::uint8_t)(XK_Delete & 0xff)] = eKey::kDelete;
    key_map[(std::uint8_t)(XK_BackSpace & 0xff)] = eKey::kBackspace;
    key_map[(std::uint8_t)(XK_Return & 0xff)] = eKey::kEnter;
    key_map[(std::uint8_t)(XK_Escape & 0xff)] = eKey::kEscape;

    input_t input[2];
    input[0] = input_t{};
    input[0].screen_size = { boot_width, boot_height };
    input[1] = input[0];

    glXMakeCurrent(display, 0, 0);

    glXMakeCurrent(display, window, glx_context);
    int i = 0;
    float frame_time = 0.f;
    float last_frame_time = 0.f;
    bool hide_cursor = false;

    for(bool run = true; run;)
    {
        using StdClock = std::chrono::high_resolution_clock;

        auto start = StdClock::now();

        XEvent xevent;
        while (XCheckWindowEvent(display, window, kEventMask, &xevent))
        {
            switch(xevent.type)
            {
            case ConfigureNotify:
            {
                XConfigureEvent const& xcevent = xevent.xconfigure;
                input[0].screen_size = std::array<int, 2>{ xcevent.width, xcevent.height };
            } break;

            case ButtonPress:
            case ButtonRelease:
            {
                //XButtonEvent const& xbevent = xevent.xbutton;
                input[0].mouse_down = (xevent.type == ButtonPress);
            } break;

            case KeyPress:
            case KeyRelease:
            {
                XKeyEvent& xkevent = xevent.xkey;
                {
                    unsigned mod_mask = 0;
                    {
                        Window a, b; int c, d, e, f;
                        XQueryPointer(display, window, &a, &b, &c, &d, &e, &f, &mod_mask);
                    }

                    char kc = '\0';
                    KeySym ks;
                    XLookupString(&xkevent, &kc, 1, &ks, nullptr);

                    if (ks == XK_Escape && xevent.type == KeyPress)
                    {
                        if (!hide_cursor)
                        {
                            XFixesHideCursor(display, window);
                            std::cout << "cursor hidden" << std::endl;
                        }
                        else
                        {
                            XFixesShowCursor(display, window);
                            std::cout << "cursor shown" << std::endl;
                        }
                        hide_cursor = !hide_cursor;
                    }

                    std::uint32_t km = 0u;
                    km |= (mod_mask & ControlMask) ? fKeyMod::kCtrl : 0u;
                    km |= (mod_mask & ShiftMask) ? fKeyMod::kShift : 0u;
                    km |= (mod_mask & Mod1Mask) ? fKeyMod::kAlt : 0u;
                    input[0].mod_down = km;

                    std::uint32_t key = (((unsigned)ks & 0xff00) == 0xff00)
                        ? (std::uint32_t)ks
                        : (std::uint32_t)kc;

                    if (key < eKey::kASCIIBegin || key >= eKey::kASCIIEnd)
                        key = key_map[key & 0xff];
                    else
                        key = (std::uint32_t)std::tolower((unsigned char)key);

                    input[0].key_down[key] = (xevent.type == KeyPress);
                }
            } break;

            case MotionNotify:
            {
                XMotionEvent const& xmevent = xevent.xmotion;
                input[0].mouse_pos = { xmevent.x, input[0].screen_size[1] - xmevent.y };
            } break;

            case DestroyNotify:
            {
                XDestroyWindowEvent const& xdwevent = xevent.xdestroywindow;
                std::cout << "window destroy" << std::endl;
                run = !(xdwevent.display == display && xdwevent.window == window);
            } break;
            default: break;
            }
        }

        if (XCheckTypedWindowEvent(display, window, ClientMessage, &xevent))
        {
            std::cout << "Client message" << std::endl;
            std::cout << XGetAtomName(display, xevent.xclient.message_type) << std::endl;
            run = !(xevent.xclient.data.l[0] == wm_delete_window);
        }
        if (!run) break;

        if (UpdateEngineModule(engine_main))
            engine_main.onreload_cb(engine);

        if (engine_main.hlib)
        {
            input[0].mouse_delta = numtk::vec2i_sub(input[0].mouse_pos, input[1].mouse_pos);

            if (hide_cursor)
            {
                numtk::Vec2i_t warp_pos = numtk::vec2i_int_div(input[0].screen_size, 2);
                if (!numtk::vec2i_equal(input[0].mouse_pos, warp_pos))
                {
                    XWarpPointer(display, None, window, 0, 0, 0, 0,
                                 warp_pos[0],
                                 input[0].screen_size[1] - warp_pos[1]);
                }
                input[0].mouse_pos = warp_pos;
            }
            else
            {
                input[0].mouse_delta = numtk::Vec2i_t{ 0, 0 };
            }

            engine_main.run_frame_cb(engine, input);
        }
        input[0].back_key_down = input[0].key_down;
        input[0].back_mod_down = input[0].mod_down;
        input[1] = input[0];

        glXSwapBuffers(display, window);

        auto end = StdClock::now();
        float measured_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
        frame_time += measured_time;
        last_frame_time = measured_time / 1000.f;
        static int const kFrameInterval = 0xff;
        i = (i + 1) & kFrameInterval;
        if (!i)
        {
            std::cout << "avg frame_time: " << (frame_time / float(kFrameInterval)) << " ms" << std::endl;
            frame_time = 0.f;
        }
    }
    glXMakeCurrent(display, 0, 0);

    engine_main.shutdown_cb(engine);

    glXDestroyContext(display, glx_context);
    XDestroyWindow(display, window);
    XCloseDisplay(display);
    return 0;
}

bool UpdateEngineModule(engine_module_t& _module)
{
    bool result = false;

    struct stat buf;
    stat(_module.path, &buf);
    int fd = open(_module.path, O_RDONLY);
    int locked = flock(fd, LOCK_EX | LOCK_NB);

    if (_module.last_load_time < buf.st_mtime && !locked && buf.st_size)
    {
        std::cout << "Load attempt..." << std::endl;

        if (syncfs(fd))
            std::cout << "syncfs failed" << std::endl;

        if (_module.hlib)
        {
            dlclose(_module.hlib);
            _module.hlib = nullptr;
        }

        _module.hlib = dlopen(_module.path, RTLD_LAZY);
        if (!_module.hlib)
            std::cout << "Library failed to reload : " << dlerror() << std::endl;
        else
            _module.last_load_time = buf.st_mtime;

        result = true;
    }

    if (locked) flock(fd, LOCK_UN | LOCK_NB);
    close(fd);

    if (_module.hlib)
    {
        // load funcs
        _module.run_frame_cb = (void(*)(void*, void*))dlsym(_module.hlib, "EngineRunFrame");
        if (!_module.run_frame_cb)
            std::cout << "EngineRunFrame not found" << std::endl;

        _module.init_cb = (void(*)(void**))dlsym(_module.hlib, "EngineInit");
        if (!_module.init_cb)
            std::cout << "EngineInit not found" << std::endl;

        _module.shutdown_cb = (void(*)(void*))dlsym(_module.hlib, "EngineShutdown");
        if (!_module.shutdown_cb)
            std::cout << "EngineShutdown not found" << std::endl;

        _module.onreload_cb = (void(*)(void*))dlsym(_module.hlib, "EngineReload");
        if (!_module.onreload_cb)
            std::cout << "EngineReload not found" << std::endl;
    }

    return result && _module.hlib;
}
