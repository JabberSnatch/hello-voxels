/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * Samuel Bourasseau wrote this file. You can do whatever you want with this
 * stuff. If we meet some day, and you think this stuff is worth it, you can
 * buy me a beer in return.
 * ----------------------------------------------------------------------------
 */

#include <array>
#include <set>
#include <vector>

#include "numtk.h"
#include "quick_vdb.hpp"


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
    numtk::dQuat_t camera_transform{ 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    bool camera_enable_mouse_control = false;
};


// [ ] VDB
// [ ] VDB -> mesh
// [ ] mesh render
// [ ] camera control


struct engine_t
{
    static constexpr unsigned kLog2ChunkSize = 4u;
    static constexpr int kChunkLoadRadius = 1;
    using VDB_t = quick_vdb::RootNode<quick_vdb::LeafNode<kLog2ChunkSize>>;

    VDB_t vdb;
    numtk::Vec3i64_t eye_position;

    std::set<numtk::Vec3i64_t> loaded_chunks;

    bool render_data_clean;
    std::vector<numtk::Vec3i64_t> rendered_surface;
};

numtk::Vec3i64_t ComputeChunkCoordinates(numtk::Vec3i64_t const& _p)
{
    return numtk::Vec3i64_t{ _p[0] >> engine_t::kLog2ChunkSize, _p[1] >> engine_t::kLog2ChunkSize, _p[2] >> engine_t::kLog2ChunkSize };
}

#include <iostream>

extern "C"
{

void EngineInit(engine_t** oEngine)
{
    *oEngine = new engine_t{
        {},
        { 0, 1, 0 },
        {},
        false,
        {}
    };
}

void EngineShutdown(engine_t* ioEngine)
{
    delete ioEngine;
}

void EngineRunFrame(engine_t* ioEngine, input_t*)
{
    numtk::Vec3i64_t const chunk = ComputeChunkCoordinates(ioEngine->eye_position);

    {
        numtk::Vec3i64_t load_bounds_min =
            numtk::vec3i64_add(chunk, { -engine_t::kChunkLoadRadius, -engine_t::kChunkLoadRadius, -engine_t::kChunkLoadRadius });

        for (std::int64_t z = 0; z <= engine_t::kChunkLoadRadius*2; ++z)
        {
            for (std::int64_t y = 0; y <= engine_t::kChunkLoadRadius*2; ++y)
            {
                for (std::int64_t x = 0; x <= engine_t::kChunkLoadRadius*2; ++x)
                {
                    numtk::Vec3i64_t chunk_index{ x, y, z };
                    numtk::Vec3i64_t chunk_world = numtk::vec3i64_add(load_bounds_min, chunk_index);

                    #if 1

                    if (!ioEngine->loaded_chunks.count(chunk_world))
                    {
                        std::int64_t chunk_size = 1 << engine_t::kLog2ChunkSize;
                        numtk::Vec3i64_t chunk_voxel_base = numtk::vec3i64_int_mul(chunk_world, chunk_size);
                        for (std::int64_t vz = 0; vz < chunk_size; ++vz)
                        {
                            for (std::int64_t vy = 0; vy < chunk_size; ++vy)
                            {
                                for (std::int64_t vx = 0; vx < chunk_size; ++vx)
                                {
                                    numtk::Vec3i64_t voxel_index{ vx, vy, vz };
                                    numtk::Vec3i64_t voxel_world = numtk::vec3i64_add(chunk_voxel_base, voxel_index);

                                    bool set_voxel = [](numtk::Vec3i64_t const& voxel_world)
                                    {
                                        return voxel_world[1] < 0;
                                        numtk::Vec3_t fvoxel_world{
                                            (float)voxel_world[0],
                                            (float)voxel_world[1],
                                            (float)voxel_world[2]
                                        };

                                        float radius = numtk::vec3_norm({ fvoxel_world[0], 0.f, fvoxel_world[2] });
                                        return (std::sin(radius) * 5.f) > fvoxel_world[1];
                                    }(voxel_world);

                                    ioEngine->vdb.set(voxel_world, set_voxel);
                                    if (set_voxel)
                                        std::cout << "Set voxel " << voxel_world[0] << " " << voxel_world[1] << " " << voxel_world[2] << std::endl;
                                }
                            }
                        }

                        ioEngine->loaded_chunks.insert(chunk_world);
                        std::cout << "Loaded " << chunk_world[0] << " " << chunk_world[1] << " " << chunk_world[2] << std::endl;
                    }

                    #else

                    ioEngine->loaded_chunks.clear();

                    #endif
                }
            }
        }
    }

    if (!ioEngine->render_data_clean)
    {
        std::int64_t max_distance = engine_t::kChunkLoadRadius * (1 << engine_t::kLog2ChunkSize);

        numtk::Vec3i64_t camera_voxel = ioEngine->eye_position;
        std::vector<numtk::Vec3i64_t> backlog{ camera_voxel };
        std::set<numtk::Vec3i64_t> processed{ camera_voxel };
        std::vector<numtk::Vec3i64_t> isosurface{};
        while (!backlog.empty())
        {
            numtk::Vec3i64_t voxel = backlog.back();
            backlog.pop_back();

            //std::cout << "Voxel " << voxel[0] << " " << voxel[1] << " " << voxel[2];

            if (ioEngine->vdb.get(voxel))
            {
                //std::cout << " isosurface" << std::endl;

                isosurface.push_back(voxel);
            }
            else
            {
                //std::cout << " outside" << std::endl;

                std::array<numtk::Vec3i64_t, 6> const neighbours{
                    numtk::vec3i64_add(voxel, {1, 0, 0}),
                    numtk::vec3i64_add(voxel, {-1, 0, 0}),
                    numtk::vec3i64_add(voxel, {0, 1, 0}),
                    numtk::vec3i64_add(voxel, {0, -1, 0}),
                    numtk::vec3i64_add(voxel, {0, 0, 1}),
                    numtk::vec3i64_add(voxel, {0, 0, -1})
                };

                for (numtk::Vec3i64_t const& neighbour : neighbours)
                {
                    numtk::Vec3i64_t neighbour_chunk = ComputeChunkCoordinates(neighbour);

                    if (std::abs(neighbour_chunk[0] - chunk[0]) <= engine_t::kChunkLoadRadius
                        && std::abs(neighbour_chunk[1] - chunk[1]) <= engine_t::kChunkLoadRadius
                        && std::abs(neighbour_chunk[2] - chunk[2]) <= engine_t::kChunkLoadRadius
                        && !processed.count(neighbour)
                    )
                    {
                        backlog.push_back(neighbour);
                        processed.insert(neighbour);
                    }
                }
            }
        }

#if 1
        std::cout << "Isosurface" << std::endl;
        for (numtk::Vec3i64_t const& voxel : isosurface)
            std::cout << voxel[0] << " " << voxel[1] << " " << voxel[2] << std::endl;
#endif

        ioEngine->rendered_surface = std::move(isosurface);
        ioEngine->render_data_clean = true;
    }

    {
        // vertex buffer
        // geometry shader
        // draw call
    }
}


}
