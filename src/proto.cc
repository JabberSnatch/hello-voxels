/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * Samuel Bourasseau wrote this file. You can do whatever you want with this
 * stuff. If we meet some day, and you think this stuff is worth it, you can
 * buy me a beer in return.
 * ----------------------------------------------------------------------------
 */

#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#include "numtk.h"
#include "quick_vdb.hpp"

#include "oglbase/handle.h"
#include "oglbase/shader.h"

#include "GL/glew.h"
#include "GL/gl.h"

#include "input.h"

struct engine_t
{
    numtk::Vec3_t camera_euler{ 0.f, 0.f, 0.f };
    numtk::Quat_t camera_current{};
    numtk::Mat4_t projection_matrix = numtk::mat4_id();

    numtk::Vec3_t player_position{ 0.f, 15.f, 0.f };
    numtk::Vec3_t player_velocity{ 0.f, -5.f, 0.f };
    bool player_on_ground = false;
    numtk::Vec3i64_t last_voxel{};
    numtk::Vec3i64_t look_at_voxel{};

    static constexpr unsigned kLog2ChunkSize = 4u;
    static constexpr unsigned kChunkSize = 1u << kLog2ChunkSize;
    static constexpr int kChunkLoadRadius = 2;
    static constexpr float kVoxelScale = .25f;
    using VDB_t =
        quick_vdb::RootNode<quick_vdb::BranchNode<quick_vdb::LeafNode<kLog2ChunkSize / 2>, kLog2ChunkSize / 2>>;
    VDB_t vdb = {};
    numtk::Vec3i64_t eye_position = { 0, 32, 0 };

    std::set<numtk::Vec3i64_t> loaded_chunks{};

    bool render_data_clean = false;
    std::vector<numtk::Vec3_t> points{};

    oglbase::BufferPtr vbo{};
    oglbase::VAOPtr vao{};
    oglbase::ProgramPtr shader_program{};
};

numtk::Vec3i64_t ComputeChunkCoordinates(numtk::Vec3i64_t const& _p)
{
    return numtk::Vec3i64_t{ _p[0] >> engine_t::kLog2ChunkSize, _p[1] >> engine_t::kLog2ChunkSize, _p[2] >> engine_t::kLog2ChunkSize };
}

numtk::Vec3_t WS_to_VS_float(numtk::Vec3_t const& _p)
{
    numtk::Vec3_t vs = numtk::vec3_float_mul(_p, 1.f/engine_t::kVoxelScale); //-.125 * 4 = -.5
    vs = numtk::vec3_add(vs, { 0.5f, 0.5f, 0.5f });
    return vs;

    // [0.125, 0.375] => 1.x || .125 * 4 = 1.0
    // [-0.125, 0.125] => 0.x
    // [-0.375, -0.125] => -1.x -.375 * 4 = -1.5
}

numtk::Vec3i64_t WS_to_VS(numtk::Vec3_t const& _p)
{
    numtk::Vec3_t vs = WS_to_VS_float(_p);
    return numtk::Vec3i64_t{
        (std::int64_t)std::trunc(vs[0]),
        (std::int64_t)std::trunc(vs[1]),
        (std::int64_t)std::trunc(vs[2])
    };
}

numtk::Vec3_t VS_to_WS_float(numtk::Vec3_t const& _p)
{
    numtk::Vec3_t ws = numtk::vec3_sub(_p, { 0.5f, 0.5f, 0.5f });
    ws = numtk::vec3_float_mul(ws, engine_t::kVoxelScale);
    return ws;
}

numtk::Vec3_t VS_to_WS(numtk::Vec3i64_t const& _p)
{
    numtk::Vec3_t vs{ (float)_p[0], (float)_p[1], (float)_p[2] };
    return VS_to_WS_float(vs);
}

void RefreshRenderBuffer(engine_t* ioEngine)
{
    numtk::Vec3i64_t const chunk = ComputeChunkCoordinates(ioEngine->eye_position);

    {
        using StdClock = std::chrono::high_resolution_clock;
        auto start = StdClock::now();

        numtk::Vec3i64_t camera_voxel = ioEngine->eye_position;
        std::vector<numtk::Vec3i64_t> backlog{ camera_voxel };
        {
            unsigned render_distance = engine_t::kChunkSize * (engine_t::kChunkLoadRadius*2+1);
            std::int64_t reserve_size = render_distance * render_distance * render_distance;
            std::cout << "Preallocating " << reserve_size * 8u * 3u << " bytes" << std::endl;
            backlog.reserve(reserve_size);
        }

        std::set<numtk::Vec3i64_t> processed{ camera_voxel };

        std::vector<numtk::Vec3_t> isosurface{};
        {
            unsigned render_distance = engine_t::kChunkSize * (engine_t::kChunkLoadRadius*2+1);
            std::int64_t reserve_size = render_distance * render_distance  * 2u;
            std::cout << "Preallocating " << reserve_size * 4u * 3u << " bytes" << std::endl;
            isosurface.reserve(reserve_size);
        }

        float dVoxelLookupTime = 0.f;

        while (!backlog.empty())
        {
            numtk::Vec3i64_t voxel = backlog.back();
            backlog.pop_back();

            //std::cout << "Voxel " << voxel[0] << " " << voxel[1] << " " << voxel[2];

            bool voxel_set = false;

            {
                using StdClock = std::chrono::high_resolution_clock;
                auto start = StdClock::now();

                voxel_set = ioEngine->vdb.get(voxel);

                auto end = StdClock::now();
                dVoxelLookupTime +=
                    static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
            }


            if (voxel_set)
            {
                //std::cout << " isosurface" << std::endl;

                isosurface.push_back({
                    (float)voxel[0],
                    (float)voxel[1],
                    (float)voxel[2]
                });
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

        ioEngine->points = std::move(isosurface);

        auto end = StdClock::now();
        float load_time =
            static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
        std::cout << "Flood fill " << load_time/1000.f << " s" << std::endl;
        std::cout << "Lookup " << dVoxelLookupTime/1000000.f << std::endl;
    }

    void* data = ioEngine->points.data();
    int point_count = ioEngine->points.size();

    ioEngine->vao.reset(0u);
    glGenVertexArrays(1, ioEngine->vao.get());
    ioEngine->vbo.reset(0u);
    glGenBuffers(1, ioEngine->vbo.get());

    glBindBuffer(GL_ARRAY_BUFFER, ioEngine->vbo);
    glBufferData(GL_ARRAY_BUFFER, point_count * 3 * sizeof(float),
                 data, GL_STATIC_DRAW);

    glBindVertexArray(ioEngine->vao);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid const*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0u);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

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

void EngineReload(engine_t* ioEngine)
{
    std::cout << "On reload" << std::endl;

    ioEngine->vdb.clear();
    ioEngine->loaded_chunks.clear();

    numtk::Vec3i64_t const chunk = ComputeChunkCoordinates(ioEngine->eye_position);

    {
        using StdClock = std::chrono::high_resolution_clock;
        auto start = StdClock::now();

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
                                        return (std::abs(voxel_world[0]) % 8 < 4
                                                && std::abs(voxel_world[2]) % 8 < 4
                                                && voxel_world[1] < (16 / ((std::abs(voxel_world[0] / 8) % 8) + 1)))
                                            || (voxel_world[1] < 1);
                                        //return (voxel_world[1] < 1);

                                        numtk::Vec3_t fvoxel_world = VS_to_WS(voxel_world);

                                        fvoxel_world =
                                            numtk::vec3_add(fvoxel_world,
                                                            numtk::vec3_constant(0.5f * engine_t::kVoxelScale));

                                        float radius = numtk::vec3_norm({ fvoxel_world[0], 0.f, fvoxel_world[2] });

                                        return (std::cos(radius*0.2f) * 2.f * (radius*0.01f+1.f)) - 4.f > fvoxel_world[1];
                                    }(voxel_world);

                                    ioEngine->vdb.set(voxel_world, set_voxel);

#if 0
                                    if (set_voxel)
                                        std::cout << "Set voxel " << voxel_world[0] << " " << voxel_world[1] << " " << voxel_world[2] << std::endl;
#endif
                                }
                            }
                        }

                        ioEngine->loaded_chunks.insert(chunk_world);
                        //std::cout << "Loaded " << chunk_world[0] << " " << chunk_world[1] << " " << chunk_world[2] << std::endl;
                    }

                }
            }
        }

        auto end = StdClock::now();
        float load_time =
            static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
        std::cout << "Voxel generation " << load_time/1000.f << " s" << std::endl;
    } // VOXEL GENERATION

    RefreshRenderBuffer(ioEngine);

    {
        static oglbase::ShaderSources_t const vshader{ "#version 330 core\n", R"__lstr__(
            in vec3 in_position;
            uniform float iExtent;
            void main()
            {
                gl_Position = vec4(in_position * iExtent, 1.0);
            }
        )__lstr__" };

        static oglbase::ShaderSources_t const fshader{ "#version 330 core\n", R"__lstr__(
            layout(location = 0) out vec4 frag_color;
            flat in vec3 outgs_voxelIndex;
            uniform vec3 iLookAtVoxel;
            void main()
            {
                vec3 color = clamp(outgs_voxelIndex / 32.0, -1.0, 1.0);
                color = color * 0.5 + vec3(0.5);
                //color = floor(mod(outgs_voxelIndex, vec3(2.0, 2.0, 2.0)));
                if (outgs_voxelIndex[0] == 0.f || outgs_voxelIndex[2] == 0.f)// || outgs_voxelIndex[1] == 0.f)
                    color = vec3(1.0, 0.0, 0.0);
                if (iLookAtVoxel == outgs_voxelIndex)
                    color = vec3(0.2, 0.2, 0.2);
                frag_color = vec4(color, 1.0);
            }
        )__lstr__" };

        static oglbase::ShaderSources_t const gshader{ "#version 330 core\n",
            "layout(points) in;\n",
            "layout(triangle_strip, max_vertices = 24) out;\n",
            "uniform mat4 iProjMat;\n"
            "uniform float iExtent;\n"
            "flat out vec3 outgs_voxelIndex;\n",
            "void main() {\n",
            "float kBaseExtent = iExtent/2.f;\n",
            "vec4 in_position = gl_in[0].gl_Position;",

            "outgs_voxelIndex = in_position.xyz / iExtent;",

            "gl_Position = iProjMat * (in_position + vec4(1, -1, 1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(-1, -1, 1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(1, -1, -1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(-1, -1, -1, 0) * kBaseExtent); EmitVertex();",
            "EndPrimitive();",

            "gl_Position = iProjMat * (in_position + vec4(1, -1, -1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(1, 1, -1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(1, -1, 1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(1, 1, 1, 0) * kBaseExtent); EmitVertex();",
            "EndPrimitive();",

            "gl_Position = iProjMat * (in_position + vec4(1, -1, 1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(1, 1, 1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(-1, -1, 1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(-1, 1, 1, 0) * kBaseExtent); EmitVertex();",
            "EndPrimitive();",

            "gl_Position = iProjMat * (in_position + vec4(-1, -1, 1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(-1, 1, 1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(-1, -1, -1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(-1, 1, -1, 0) * kBaseExtent); EmitVertex();",
            "EndPrimitive();",

            "gl_Position = iProjMat * (in_position + vec4(-1, -1, -1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(-1, 1, -1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(1, -1, -1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(1, 1, -1, 0) * kBaseExtent); EmitVertex();",
            "EndPrimitive();",

            "gl_Position = iProjMat * (in_position + vec4(1, 1, -1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(-1, 1, -1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(1, 1, 1, 0) * kBaseExtent); EmitVertex();",
            "gl_Position = iProjMat * (in_position + vec4(-1, 1, 1, 0) * kBaseExtent); EmitVertex();",
            "EndPrimitive();",
            "}"
        };

        std::string log{};
        oglbase::ShaderPtr vbin = oglbase::CompileShader(GL_VERTEX_SHADER, vshader, &log);
        std::cout << "Vshader " << log << std::endl;
        oglbase::ShaderPtr fbin = oglbase::CompileShader(GL_FRAGMENT_SHADER, fshader, &log);
        std::cout << "Fshader " << log << std::endl;
        oglbase::ShaderPtr gbin = oglbase::CompileShader(GL_GEOMETRY_SHADER, gshader);
        std::cout << "Gshader " << log << std::endl;
        ioEngine->shader_program = oglbase::LinkProgram({ vbin, fbin, gbin }, &log);
        std::cout << "Link " << log << std::endl;
    } // RENDERDATA
}

void EngineRunFrame(engine_t* ioEngine, input_t const* iInput)
{
    static constexpr float kFrameTime = 0.016f;

#if 0
    std::cout << iInput->key_down[' '] << " "
              << ((iInput->mod_down & fKeyMod::kShift) != 0) << std::endl
              << iInput->key_down['w'] << " "
              << iInput->key_down['a'] << " "
              << iInput->key_down['s'] << " "
              << iInput->key_down['d'] << std::endl;
#endif

    {

        if (iInput->key_down['q'])
        {
            ioEngine->player_position = { 0.f, 15.f, 0.f };
            ioEngine->player_velocity = { 0.f, -5.f, 0.f };
            ioEngine->player_on_ground = false;
        }

        {
            static constexpr float kAngleSpeed = 0.5f;
            numtk::Vec3_t rotation_delta = numtk::vec3_float_mul(numtk::Vec3_t{
                -(float)iInput->mouse_delta[1],
                -(float)iInput->mouse_delta[0],
                0.f }
            , kAngleSpeed * (3.1415926536f / 180.f));

            ioEngine->camera_euler = numtk::vec3_add(ioEngine->camera_euler, rotation_delta);
        }
    }

    {
        numtk::Quat_t target = numtk::quat_from_euler(ioEngine->camera_euler);
        ioEngine->camera_current = numtk::quat_normalise(numtk::quat_slerp(ioEngine->camera_current, target, 0.2f));
    }

    {
        {
            numtk::Vec3_t d{ 0.f, 0.f, 0.f };

            if (iInput->key_down['w'])
                d = numtk::vec3_add(d, { 0.f, 0.f, 1.f });
            if (iInput->key_down['s'])
                d = numtk::vec3_add(d, { 0.f, 0.f, -1.f });
            if (iInput->key_down['a'])
                d = numtk::vec3_add(d, { 1.f, 0.f, 0.f });
            if (iInput->key_down['d'])
                d = numtk::vec3_add(d, { -1.f, 0.f, 0.f });

            if (numtk::vec3_dot(d, d) > 0.f)
            {
                d = numtk::vec3_normalise(d);

                static constexpr float kLineSpeed = 5.0f;
                numtk::Vec3_t dp = d;
                numtk::Mat4_t camrot = numtk::mat4_from_quat(ioEngine->camera_current);
                numtk::Vec3_t wsdp = numtk::vec3_from_vec4(numtk::mat4_vec4_mul(camrot, numtk::vec3_float_concat(dp, 0.f)));
                wsdp[1] = 0.f;
                wsdp = numtk::vec3_normalise(wsdp);
                wsdp = numtk::vec3_float_mul(wsdp, kLineSpeed * kFrameTime);

                ioEngine->player_position = numtk::vec3_add(ioEngine->player_position, wsdp);
            }

            if (iInput->key_down[' '])
            {
                quick_vdb::Position_t voxel{ ioEngine->look_at_voxel[0], ioEngine->look_at_voxel[1] + 1, ioEngine->look_at_voxel[2] };
                ioEngine->vdb.set(voxel);
                RefreshRenderBuffer(ioEngine);
            }
        }

        if (!ioEngine->player_on_ground)
        {
            static const numtk::Vec3_t g = { 0.f, -9.8f, 0.f };

            ioEngine->player_velocity = numtk::vec3_add(
                numtk::vec3_float_mul(g, kFrameTime),
                ioEngine->player_velocity
            );

            ioEngine->player_position = numtk::vec3_add(
                numtk::vec3_float_mul(ioEngine->player_velocity, kFrameTime),
                ioEngine->player_position
            );
        }

        {

            quick_vdb::Position_t vp = WS_to_VS(ioEngine->player_position);

            if (ioEngine->last_voxel != vp)
            {
                if (ioEngine->vdb.get(vp))
                {
                    ioEngine->player_on_ground = true;

                    static constexpr unsigned kMaxIterationCount = 2048u;
                    float offset = 0.f;
                    for (unsigned it = 0u;
                         ioEngine->vdb.get(vp) && it < kMaxIterationCount; ++it)
                    {
                        ++vp[1];
                        offset += 1.f;
                    }

                    ioEngine->player_position[1] += (offset-1.f) * engine_t::kVoxelScale;
                    ioEngine->player_velocity = numtk::Vec3_t{ 0.f, 0.f, 0.f };
                }
                else
                {
                    numtk::Vec3i64_t t0 = vp;
                    --t0[1];
                    if (!ioEngine->vdb.get(t0))
                    {
                        ioEngine->player_on_ground = false;
                    }
                }
            }
        }

        {
            ioEngine->last_voxel = WS_to_VS(ioEngine->player_position);
        }
    }

    {
        numtk::Mat4_t camrot = numtk::mat4_from_quat(ioEngine->camera_current);
        numtk::Vec4_t camforward = numtk::mat4_vec4_mul(camrot, numtk::Vec4_t{ 0.f, 0.f, -1.f, 0.f});
        numtk::Vec4_t camup = numtk::mat4_vec4_mul(camrot, numtk::Vec4_t{ 0.f, 1.f, 0.f, 0.f });
        numtk::Vec3_t campos = numtk::vec3_add(ioEngine->player_position, { 0.f, 1.7f, 0.f });

        numtk::Mat4_t viewmat = numtk::mat4_view(
            numtk::vec3_from_vec4(camforward),
            numtk::vec3_from_vec4(camup),
            campos
        );

        float aspect_ratio = (float)iInput->screen_size[1] / (float)iInput->screen_size[0];
        ioEngine->projection_matrix = numtk::mat4_mul(
            numtk::mat4_perspective(0.01f, 1000.f, 3.1415926534f*0.5f, aspect_ratio),
            viewmat
        );


        {
            numtk::Vec3_t rd = numtk::vec3_from_vec4(numtk::mat4_vec4_mul(camrot, numtk::Vec4_t{ 0.f, 0.f, 1.f, 0.f }));
            numtk::Vec3_t invrd = numtk::Vec3_t{ 1.f/rd[0], 1.f/rd[1], 1.f/rd[2] };

            numtk::Vec3_t p = campos;
            quick_vdb::Position_t vp = WS_to_VS(p);
            p = WS_to_VS_float(p);

#if 0
            std::cout << "last voxel " << ioEngine->last_voxel[0] << " " << ioEngine->last_voxel[1] << " " << ioEngine->last_voxel[2] << std::endl;
            std::cout << "ro " << p[0] << " " << p[1] << " " << p[2] << std::endl;
            std::cout << "rd " << rd[0] << " " << rd[1] << " " << rd[2] << std::endl;
#endif

            bool hit = false;
            for(;;)
            {
                if (numtk::vec3_norm(numtk::vec3_sub(VS_to_WS_float(p), campos)) > 5.f)
                    break;
                if (ioEngine->vdb.get(vp)) {
                    //std::cout << "Looking at " << vp[0] << " " << vp[1] << " " << vp[2] << std::endl;
                    hit = true; break; }

                numtk::Vec3_t fract{ std::fmod(p[0], 1.f), std::fmod(p[1], 1.f), std::fmod(p[2], 1.f) };

                numtk::Vec3_t t0{
                    (float)vp[0] - p[0] - ((vp[0] < 0) ? 1.f : 0.f),
                    (float)vp[1] - p[1] - ((vp[1] < 0) ? 1.f : 0.f),
                    (float)vp[2] - p[2] - ((vp[2] < 0) ? 1.f : 0.f)
                };
                t0 = numtk::vec3_cwise_mul(t0, invrd);

                numtk::Vec3_t t1{
                    (float)vp[0] - p[0] + ((vp[0] >= 0) ? 1.f : 0.f),
                    (float)vp[1] - p[1] + ((vp[1] >= 0) ? 1.f : 0.f),
                    (float)vp[2] - p[2] + ((vp[2] >= 0) ? 1.f : 0.f)
                };
                t1 = numtk::vec3_cwise_mul(t1, invrd);

                numtk::Vec3_t cwmin{ std::min(t0[0], t1[0]), std::min(t0[1], t1[1]), std::min(t0[2], t1[2]) };
                numtk::Vec3_t cwmax{ std::max(t0[0], t1[0]), std::max(t0[1], t1[1]), std::max(t0[2], t1[2]) };

                float tmin = std::numeric_limits<float>::infinity();
                int mincomp = 0;
                if (cwmax[0] < tmin && cwmax[0] > 0.f) tmin = cwmax[0];
                if (cwmax[1] < tmin && cwmax[1] > 0.f){ tmin = cwmax[1]; mincomp = 1; }
                if (cwmax[2] < tmin && cwmax[2] > 0.f){ tmin = cwmax[2]; mincomp = 2; }
                if (cwmin[0] < tmin && cwmin[0] > 0.f){ tmin = cwmin[0]; mincomp = 0; }
                if (cwmin[1] < tmin && cwmin[1] > 0.f){ tmin = cwmin[1]; mincomp = 1; }
                if (cwmin[2] < tmin && cwmin[2] > 0.f){ tmin = cwmin[2]; mincomp = 2; }

                p = numtk::vec3_add(p, numtk::vec3_float_mul(rd, tmin * 1.0f));
                vp[mincomp] += (rd[mincomp] > 0.f) ? 1 : -1;
            }

            if (hit)
                ioEngine->look_at_voxel = vp;
            else
                ioEngine->look_at_voxel = { -1337, -1337, -1337 };
        }
    }

    {
        // draw call
        glViewport(0, 0, iInput->screen_size[0], iInput->screen_size[1]);
        glDisable(GL_MULTISAMPLE);
        glDisable(GL_BLEND);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_STENCIL_TEST);
        glEnable(GL_DEPTH_TEST);
        glFrontFace(GL_CCW);
        glEnable(GL_CULL_FACE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        static GLfloat const clear_color[]{ 0.5f, 0.5f, 0.5f, 1.f };
        glClearBufferfv(GL_COLOR, 0, clear_color);
        static GLfloat const clear_depth = 1.f;
        glClearBufferfv(GL_DEPTH, 0, &clear_depth);

        glUseProgram(ioEngine->shader_program);
        {
            int const projmat_loc = glGetUniformLocation(ioEngine->shader_program, "iProjMat");
            if (projmat_loc >= 0)
                glUniformMatrix4fv(projmat_loc, 1, GL_FALSE, &ioEngine->projection_matrix[0]);

            int const extent_loc = glGetUniformLocation(ioEngine->shader_program, "iExtent");
            if (extent_loc >= 0)
                glUniform1f(extent_loc, engine_t::kVoxelScale);

            numtk::Vec3_t v{
                (float)ioEngine->look_at_voxel[0],
                (float)ioEngine->look_at_voxel[1],
                (float)ioEngine->look_at_voxel[2]
            };
            int const lookatvoxel_loc = glGetUniformLocation(ioEngine->shader_program, "iLookAtVoxel");
            if (lookatvoxel_loc >= 0)
                glUniform3fv(lookatvoxel_loc, 1, &v[0]);
        }

        glBindVertexArray(ioEngine->vao);
        glDrawArrays(GL_POINTS, 0, ioEngine->points.size());
        glBindVertexArray(0u);

        glUseProgram(0u);
    }
}


}
