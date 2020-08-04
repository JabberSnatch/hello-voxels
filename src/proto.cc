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

#include "oglbase/handle.h"
#include "oglbase/shader.h"

#include "GL/glew.h"
#include "GL/gl.h"


numtk::Mat4_t
MakeCameraMatrix(numtk::Vec3_t const& _p, numtk::Vec3_t const& _t, numtk::Vec3_t const& _u)
{
    numtk::Vec3_t const f = numtk::vec3_normalise(numtk::vec3_sub(_p, _t));
    numtk::Vec3_t const r = numtk::vec3_normalise(numtk::vec3_cross(_u, f));
    numtk::Vec3_t const u = numtk::vec3_cross(f, r);

    numtk::Vec3_t const p = numtk::vec3_float_mul(_p, -1.f);
    numtk::Vec3_t const t{ numtk::vec3_dot(p, r), numtk::vec3_dot(p, u), numtk::vec3_dot(p, f) };

    return numtk::mat4_col(numtk::Vec4_t{r[0], u[0], f[0], 0.f},
                           numtk::Vec4_t{r[1], u[1], f[1], 0.f},
                           numtk::Vec4_t{r[2], u[2], f[2], 0.f},
                           numtk::vec3_float_concat(t, 1.f));
}


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

struct engine_t
{
    numtk::Mat4_t projection_matrix;

    static constexpr unsigned kLog2ChunkSize = 4u;
    static constexpr int kChunkLoadRadius = 2;
    using VDB_t = quick_vdb::RootNode<quick_vdb::LeafNode<kLog2ChunkSize>>;
    VDB_t vdb;
    numtk::Vec3i64_t eye_position;

    std::set<numtk::Vec3i64_t> loaded_chunks;

    bool render_data_clean;
    std::vector<numtk::Vec3_t> points;

    oglbase::BufferPtr vbo;
    oglbase::VAOPtr vao;
    oglbase::ProgramPtr shader_program;
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
        numtk::mat4_id(),
        engine_t::VDB_t{},
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

void EngineReload(engine_t* ioEngine)
{
    std::cout << "On reload" << std::endl;

    ioEngine->vdb.clear();
    ioEngine->loaded_chunks.clear();

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
                                        numtk::Vec3_t fvoxel_world{
                                            (float)voxel_world[0],
                                            (float)voxel_world[1],
                                            (float)voxel_world[2]
                                        };

                                        float radius = numtk::vec3_norm({ fvoxel_world[0], 0.f, fvoxel_world[2] });

                                        return (std::sin(radius*0.5f) * 6.f) - 4.f > fvoxel_world[1];
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
    } // VOXEL GENERATION

    {
        std::int64_t max_distance = engine_t::kChunkLoadRadius * (1 << engine_t::kLog2ChunkSize);

        numtk::Vec3i64_t camera_voxel = ioEngine->eye_position;
        std::vector<numtk::Vec3i64_t> backlog{ camera_voxel };
        std::set<numtk::Vec3i64_t> processed{ camera_voxel };
        std::vector<numtk::Vec3_t> isosurface{};
        while (!backlog.empty())
        {
            numtk::Vec3i64_t voxel = backlog.back();
            backlog.pop_back();

            //std::cout << "Voxel " << voxel[0] << " " << voxel[1] << " " << voxel[2];

            if (ioEngine->vdb.get(voxel))
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

#if 0
        std::cout << "Isosurface" << std::endl;
        for (numtk::Vec3i64_t const& voxel : isosurface)
            std::cout << voxel[0] << " " << voxel[1] << " " << voxel[2] << std::endl;
#endif

        ioEngine->points = std::move(isosurface);

        // =====================================================================
        // OPENGL
        // =====================================================================

        void* data = ioEngine->points.data();
        int point_count = ioEngine->points.size();

        glGenVertexArrays(1, ioEngine->vao.get());
        glGenBuffers(1, ioEngine->vbo.get());

        glBindBuffer(GL_ARRAY_BUFFER, ioEngine->vbo);
        glBufferData(GL_ARRAY_BUFFER, point_count * 3 * sizeof(float),
                     data, GL_STATIC_DRAW);

        glBindVertexArray(ioEngine->vao);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid const*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0u);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        static oglbase::ShaderSources_t const vshader{ "#version 330 core\n", R"__lstr__(
            in vec3 in_position;
            void main()
            {
                gl_Position = vec4(in_position, 1.0);
            }
        )__lstr__" };

        static oglbase::ShaderSources_t const fshader{ "#version 330 core\n", R"__lstr__(
            layout(location = 0) out vec4 frag_color;
            flat in vec3 outgs_voxelIndex;
            void main()
            {
                vec3 color = clamp(outgs_voxelIndex / 32.0, -1.0, 1.0);
                color = color * 0.5 + vec3(0.5);
                frag_color = vec4(color, 1.0);
            }
        )__lstr__" };

        static oglbase::ShaderSources_t const gshader{ "#version 330 core\n",
            "layout(points) in;\n",
            "layout(triangle_strip, max_vertices = 24) out;\n",
            "uniform mat4 iProjMat;\n"
            "flat out vec3 outgs_voxelIndex;\n",
            "const vec4 kBaseExtent = 0.5 * vec4(1.0, 1.0, 1.0, 0.0);\n",
            "void main() {\n",
            "vec4 in_position = gl_in[0].gl_Position;",

            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(1, -1, 1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(-1, -1, 1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(1, -1, -1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(-1, -1, -1, 0) * kBaseExtent); EmitVertex();",
            "EndPrimitive();",

            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(1, -1, -1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(1, 1, -1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(1, -1, 1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(1, 1, 1, 0) * kBaseExtent); EmitVertex();",
            "EndPrimitive();",

            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(1, -1, 1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(1, 1, 1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(-1, -1, 1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(-1, 1, 1, 0) * kBaseExtent); EmitVertex();",
            "EndPrimitive();",

            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(-1, -1, 1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(-1, 1, 1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(-1, -1, -1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(-1, 1, -1, 0) * kBaseExtent); EmitVertex();",
            "EndPrimitive();",

            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(-1, -1, -1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(-1, 1, -1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(1, -1, -1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(1, 1, -1, 0) * kBaseExtent); EmitVertex();",
            "EndPrimitive();",

            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(1, 1, -1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(-1, 1, -1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
            "gl_Position = iProjMat * (in_position + vec4(1, 1, 1, 0) * kBaseExtent); EmitVertex();",
            "outgs_voxelIndex = in_position.xyz;"
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
    {
        float aspect_ratio = (float)iInput->screen_size[1] / (float)iInput->screen_size[0];
        ioEngine->projection_matrix = numtk::mat4_mul(
            numtk::perspective(0.01f, 1000.f, 3.1415926534f*0.6f, aspect_ratio),
            MakeCameraMatrix({ 0.f, 10.f, 5.f }, { 0.f, 0.f, 0.f }, { 0.f, 1.f, 0.f })
        );

        // draw call
        glViewport(0, 0, iInput->screen_size[0], iInput->screen_size[1]);
        glDisable(GL_MULTISAMPLE);
        glDisable(GL_BLEND);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_STENCIL_TEST);
        glEnable(GL_DEPTH_TEST);
        glFrontFace(GL_CCW);
        glEnable(GL_CULL_FACE);

        static GLfloat const clear_color[]{ 0.5f, 0.5f, 0.5f, 1.f };
        glClearBufferfv(GL_COLOR, 0, clear_color);
        static GLfloat const clear_depth = 1.f;
        glClearBufferfv(GL_DEPTH, 0, &clear_depth);

        glUseProgram(ioEngine->shader_program);
        {
            int const projmat_loc = glGetUniformLocation(ioEngine->shader_program, "iProjMat");
            if (projmat_loc >= 0)
                glUniformMatrix4fv(projmat_loc, 1, GL_FALSE, &ioEngine->projection_matrix[0]);
        }

        glBindVertexArray(ioEngine->vao);
        glDrawArrays(GL_POINTS, 0, ioEngine->points.size());
        glBindVertexArray(0u);

        glUseProgram(0u);
    }
}


}
