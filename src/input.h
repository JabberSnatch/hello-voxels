#pragma once

#include <array>

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
    std::array<int, 2> screen_size{ 0, 0 };
    bool mouse_down{ false };
    std::array<int, 2> mouse_pos{ -1, -1 };
    std::array<int, 2> mouse_delta{ 0, 0 };

    std::array<bool, 256> key_down;
    std::uint32_t mod_down;

    std::array<bool, 256> back_key_down;
    std::uint32_t back_mod_down;
};
