
#include <array>
#include "numtk.h"

enum eKey
{
    kSpecialBegin = 1u,
    kTab = kSpecialBegin,
    kLeft,
    kRight,
    kUp,
    kDown,
    kPageUp,
    kPageDown,
    kHome,
    kEnd,
    kInsert,
    kDelete,
    kBackspace,
    kEnter,
    kEscape,
    kSpecialEnd,

    kASCIIBegin = 0x20,
    kASCIIEnd = 0x7f,
};

enum fKeyMod
{
    kCtrl = 1u << 0,
    kShift = 1u << 1,
    kAlt = 1u << 2
};

struct input_t
{
    numtk::Vec2i_t screen_size{ 0, 0 };
    bool mouse_down{ false };
    numtk::Vec2i_t mouse_pos{ -1, -1 };
    numtk::Vec2i_t mouse_delta{ 0, 0 };
    std::array<bool, 256> key_down;
    std::uint32_t mod_down;
    std::array<eKey, 256> key_map;
};

struct state_t
{
    numtk::dQuat_t camera_transform{1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    bool camera_enable_mouse_control = false;
};


// [ ] VDB
// [ ] VDB -> mesh
// [ ] mesh render
// [ ] camera control

struct engine_t
{
};

#include <iostream>

extern "C"
{

void EngineInit(engine_t** oEngine)
{
    *oEngine = new engine_t{};
}

void EngineShutdown(engine_t* ioEngine)
{
    delete ioEngine;
}

void EngineRunFrame(engine_t*, input_t*)
{
    std::cout << "Hello World" << std::endl;
}


}
