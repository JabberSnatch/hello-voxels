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
#include <cstring>
#include <iostream>
#include <numeric>
#include <limits>
#include <set>
#include <thread>
#include <vector>

#include "numtk.h"
#define QVDB_ENABLE_CACHE
//#define QVDB_STD_BITSET
#include "quick_vdb.hpp"

#include "oglbase/error.h"
#include "oglbase/handle.h"
#include "oglbase/shader.h"

#include "GL/glew.h"
#include "GL/gl.h"

#include "input.h"
#include "timer.h"

constexpr bool kGeometryPipeline = false;
constexpr bool kFragmentPipeline = true;

extern oglbase::ShaderSources_t const rtfrag;
extern oglbase::ShaderSources_t const rtvert;


struct engine_t
{
    numtk::Vec3_t camera_euler{ 0.f, 0.f, 0.f };
    numtk::Quat_t camera_current{};
    numtk::Vec3_t campos() const { return numtk::vec3_add(player_position, { 0.f, 1.7f, 0.f }); }
    numtk::Mat4_t projection_matrix = numtk::mat4_id();
    numtk::Mat4_t projection_invmatrix = numtk::mat4_id();

    static constexpr numtk::Vec3i64_t kInvalidChunkIndex{
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::max()
    };
    numtk::Vec3i64_t campos_chunk_index{ kInvalidChunkIndex };

    numtk::Vec3_t player_position{ 0.f, 16.f, 0.f };
    numtk::Vec3_t player_velocity{ 0.f, -5.f, 0.f };
    bool noclip_mode = false;
    bool player_on_ground = false;
    numtk::Vec3i64_t last_voxel{};
    numtk::Vec3i64_t look_at_voxel{ -1337, -1337, -1337 };

    static constexpr unsigned kLog2ChunkSize = 6u;
    static constexpr unsigned kChunkSize = 1u << kLog2ChunkSize;
    static constexpr int kChunkLoadRadius = 0;
    static constexpr float kVoxelScale = .25f;
    using VDB_t = quick_vdb::RootNode< quick_vdb::BranchNode< quick_vdb::BranchNode<
                  quick_vdb::LeafNode<kLog2ChunkSize>, 4u>, 4u>>;
    VDB_t vdb = {};

    numtk::Vec3i64_t eye_position = { 0, 0, 0 };

    std::set<numtk::Vec3i64_t> generated_chunks{};

    bool render_data_clean = false;
    std::vector<numtk::Vec3_t> points{};

    oglbase::BufferPtr vbo{};
    oglbase::VAOPtr vao{};
    oglbase::ProgramPtr shader_program{};

    oglbase::BufferPtr staging_buffer{};
    oglbase::SamplerPtr sampler{};

    oglbase::TexturePtr chunk_texture{};
    GLuint64 chunk_handle{};

    struct ChunkResources
    {
        numtk::Vec3i64_t base;
        oglbase::TexturePtr texture;
        GLuint64 handle;
    };

    static constexpr int kChunkLoadExtent = kChunkLoadRadius*2+1;
    static constexpr int kChunkLoadIndex(numtk::Vec3i64_t const& _chunk_index)
    {
        return _chunk_index[0]
            + kChunkLoadExtent * (_chunk_index[1]
                                  + kChunkLoadExtent * _chunk_index[2]);
    }

    static constexpr int kChunkLoadCount = kChunkLoadExtent * kChunkLoadExtent * kChunkLoadExtent;
    std::array<ChunkResources, kChunkLoadCount> chunks{};
};

void GenerateChunk(engine_t::VDB_t& vdb, numtk::Vec3i64_t const& chunk_index);
void UploadChunk(engine_t::ChunkResources& ioResources,
                 engine_t::VDB_t& vdb,
                 oglbase::BufferPtr const& staging_buffer);


numtk::Vec3i64_t ComputeChunkIndex(numtk::Vec3i64_t const& _p)
{
    return numtk::Vec3i64_t{
        _p[0] >> engine_t::kLog2ChunkSize,
        _p[1] >> engine_t::kLog2ChunkSize,
        _p[2] >> engine_t::kLog2ChunkSize
    };
}

numtk::Vec3i64_t ChunkBaseFromIndex(numtk::Vec3i64_t const& _chunkIndex)
{
    return numtk::vec3i64_int_mul(_chunkIndex, engine_t::kChunkSize);
}

numtk::Vec3_t ComputeChunkLocalCoordinates(numtk::Vec3i64_t const& _p)
{
    float invchunksize = 1.f / float(1u << engine_t::kLog2ChunkSize);
    return numtk::Vec3_t{
        (float)(_p[0] - (_p[0] & ((1u << engine_t::kLog2ChunkSize) - 1u))) * invchunksize,
        (float)(_p[1] - (_p[1] & ((1u << engine_t::kLog2ChunkSize) - 1u))) * invchunksize,
        (float)(_p[2] - (_p[2] & ((1u << engine_t::kLog2ChunkSize) - 1u))) * invchunksize
    };
}

numtk::Vec3_t ComputeChunkLocalCoordinates(numtk::Vec3_t const& _p)
{
    float invchunksize = 1.f / float(1u << engine_t::kLog2ChunkSize);
    return numtk::Vec3_t{
        _p[0] * invchunksize,
        _p[1] * invchunksize,
        _p[2] * invchunksize
    };
}

numtk::Vec3_t WS_to_VS_float(numtk::Vec3_t const& _p)
{
    numtk::Vec3_t vs = numtk::vec3_float_mul(_p, 1.f/engine_t::kVoxelScale); //-.125 * 4 = -.5
    //vs = numtk::vec3_add(vs, { 0.5f, 0.5f, 0.5f });
    return vs;

    // [0.125, 0.375] => 1.x || .125 * 4 = 1.0
    // [-0.125, 0.125] => 0.x
    // [-0.375, -0.125] => -1.x -.375 * 4 = -1.5
}

numtk::Vec3i64_t WS_to_VS(numtk::Vec3_t const& _p)
{
    numtk::Vec3_t vs = WS_to_VS_float(_p);
    return numtk::Vec3i64_t{
        (std::int64_t)std::floor(vs[0]),
        (std::int64_t)std::floor(vs[1]),
        (std::int64_t)std::floor(vs[2])
    };
}

numtk::Vec3_t VS_to_WS_float(numtk::Vec3_t const& _p)
{
    numtk::Vec3_t ws = _p;//numtk::vec3_sub(_p, { 0.5f, 0.5f, 0.5f });
    ws = numtk::vec3_float_mul(ws, engine_t::kVoxelScale);
    return ws;
}

numtk::Vec3_t VS_to_WS(numtk::Vec3i64_t const& _p)
{
    numtk::Vec3_t vs{ (float)_p[0], (float)_p[1], (float)_p[2] };
    return VS_to_WS_float(vs);
}

void RefreshRenderBuffer(engine_t* ioEngine, numtk::Vec3i64_t const& eye_position)
{
    numtk::Vec3i64_t const chunk = ComputeChunkIndex(eye_position);

    if (kGeometryPipeline) // geometry shader pipeline
    {
        using StdClock = std::chrono::high_resolution_clock;
        auto start = StdClock::now();

        numtk::Vec3i64_t camera_voxel = eye_position;
        std::vector<numtk::Vec3i64_t> backlog{ camera_voxel };
        {
            unsigned render_distance = engine_t::kChunkSize * (engine_t::kChunkLoadExtent);
            std::int64_t reserve_size = render_distance * render_distance * render_distance;
            std::cout << "Preallocating " << reserve_size * 8u * 3u << " bytes" << std::endl;
            backlog.reserve(reserve_size);
        }

        std::set<numtk::Vec3i64_t> processed{ camera_voxel };

        std::vector<numtk::Vec3_t> isosurface{};
        {
            unsigned render_distance = engine_t::kChunkSize * (engine_t::kChunkLoadExtent);
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
                    numtk::Vec3i64_t neighbour_chunk = ComputeChunkIndex(neighbour);

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

void GenerateChunk(engine_t::VDB_t& vdb, numtk::Vec3i64_t const& chunk_base)
{
    Timer<Info> timer0("GenerateChunk");
    std::cout << "Generating chunk" << std::endl;

    numtk::Vec3i64_t const chunk_voxel_base = chunk_base;

    for (std::int64_t vz = 0; vz < engine_t::kChunkSize; ++vz)
    {
        for (std::int64_t vy = 0; vy < engine_t::kChunkSize; ++vy)
        {
            for (std::int64_t vx = 0; vx < engine_t::kChunkSize; ++vx)
            {
                numtk::Vec3i64_t voxel_index{ vx, vy, vz };
                numtk::Vec3i64_t voxel_world = numtk::vec3i64_add(chunk_voxel_base, voxel_index);

                bool set_voxel = [](numtk::Vec3i64_t const& voxel_world)
                {
                    #if 0
                    return ((voxel_world[0] & 1u)
                            & (voxel_world[1] & 1u)
                            & (voxel_world[2] & 1u)) != 0u;
                    #endif
                    //return (voxel_world[1] < 16);

                    #if 0
                    return (std::abs(voxel_world[0]) % 8 < 4
                            && std::abs(voxel_world[2]) % 8 < 4
                            && voxel_world[1] < (16 / ((std::abs(voxel_world[0] / 8) % 8) + 1)))
                        || (voxel_world[1] < 1);
                    #endif

                    numtk::Vec3_t fvoxel_world = VS_to_WS(voxel_world);

                    fvoxel_world =
                        numtk::vec3_add(fvoxel_world,
                                        numtk::vec3_constant(0.5f * engine_t::kVoxelScale));

                    float radius = numtk::vec3_norm({ fvoxel_world[0], 0.f, fvoxel_world[2] });
                    float otherradius = numtk::vec3_norm({ fvoxel_world[0] - 2.f, 0.f, fvoxel_world[2]+10.f });

                    return fvoxel_world[1] < (std::cos(radius*5.f) * 0.5f) + std::cos(otherradius*0.2f) * 4.f;
                }(voxel_world);

                vdb.set(voxel_world, set_voxel);

                #if 0
                if (set_voxel)
                    std::cout << "Set voxel " << voxel_world[0] << " " << voxel_world[1] << " " << voxel_world[2] << std::endl;
                #endif
            }
        }
    }

}

void UploadChunk(engine_t::ChunkResources& ioResources, engine_t::VDB_t& vdb, oglbase::BufferPtr const& staging_buffer)
{
    Timer<Info> timer0("UploadChunk");
    std::cout << "Uploading chunk..." << std::endl;

    std::size_t size = 0u;
    std::uint64_t const* buffer = nullptr;
    vdb.GetLeafPointer(ioResources.base, &size, &buffer);

    bool unmap_successful = false;
    unsigned char* dest = (unsigned char*)glMapNamedBuffer(staging_buffer,
                                                           GL_WRITE_ONLY);

    if (!dest)
        oglbase::PrintError(std::cout);

    if (size != -1ull)
    {
        if (size == 0)
        {
            bool value = (bool)buffer;
            std::memset(dest, value ? 0xff : 0, 1u << (engine_t::kLog2ChunkSize * 3u));
        }
        else
        {
            for (int i = 0; i < size; ++i)
            {
                std::uint64_t v = buffer[i];
                int bit_index = 0u;
                while (v)
                {
                    std::uint8_t const d = (v & 1u) ? 0xff : 0x00;
                    dest[i * 64 + bit_index] = d;
                    v >>= 1;
                    ++bit_index;
                }

                if (bit_index < 64u)
                    std::memset(&dest[i * 64 + bit_index], 0, (64 - bit_index));
            }
        }
    }
    else
        std::memset(dest, 0, 1u << (engine_t::kLog2ChunkSize * 3u));

    unmap_successful = glUnmapNamedBuffer(staging_buffer);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, staging_buffer);
    glBindTexture(GL_TEXTURE_3D, ioResources.texture);

    static const GLsizei kTextureSide = 1 << engine_t::kLog2ChunkSize;
    glTexSubImage3D(GL_TEXTURE_3D, 0,
                    0, 0, 0, kTextureSide, kTextureSide, kTextureSide,
                    GL_RED, GL_UNSIGNED_BYTE, nullptr);

    glBindTexture(GL_TEXTURE_3D, 0u);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0u);

    GLenum error_code = glGetError();
    if (error_code != GL_NO_ERROR)
        std::cout << "OpenGL error detected" << std::endl;
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
    ioEngine->generated_chunks.clear();
    for (engine_t::ChunkResources& chunk : ioEngine->chunks)
        chunk.base = { engine_t::kInvalidChunkIndex };

    if (kGeometryPipeline) // geometry shader pipeline
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
                //color = color * 0.5 + vec3(0.5);

                //color = floor(mod(outgs_voxelIndex, vec3(2.0, 2.0, 2.0)));

                if (outgs_voxelIndex[0] == 0.f || outgs_voxelIndex[2] == 0.f) // || outgs_voxelIndex[1] == 0.f)
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

    if (kFragmentPipeline)
    {
        oglbase::ShaderSources_t const vshader = rtvert;
        oglbase::ShaderSources_t const fshader = rtfrag;

        std::string log{};
        oglbase::ShaderPtr fbin = oglbase::CompileShader(GL_FRAGMENT_SHADER, fshader, &log);
        std::cout << "Fshader " << log << std::endl;
        oglbase::ShaderPtr vbin = oglbase::CompileShader(GL_VERTEX_SHADER, vshader, &log);
        std::cout << "Vshader " << log << std::endl;
        oglbase::ProgramPtr program = oglbase::LinkProgram({ fbin, vbin }, &log);
        std::cout << "Link " << log << std::endl;
        ioEngine->shader_program = std::move(program);

        ioEngine->vao.reset(0u);
        glGenVertexArrays(1, ioEngine->vao.get());

        ioEngine->staging_buffer.reset(0u);
        glCreateBuffers(1, ioEngine->staging_buffer.get());
        glNamedBufferStorage(ioEngine->staging_buffer,
                             1u << (engine_t::kLog2ChunkSize * 3u),
                             nullptr,
                             GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT | GL_CLIENT_STORAGE_BIT);

        ioEngine->sampler.reset(0u);
        glGenSamplers(1, ioEngine->sampler.get());
        glSamplerParameteri(ioEngine->sampler, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glSamplerParameteri(ioEngine->sampler, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glSamplerParameteri(ioEngine->sampler, GL_TEXTURE_WRAP_R, GL_REPEAT);
        glSamplerParameteri(ioEngine->sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glSamplerParameteri(ioEngine->sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        for (int chunkIndex = 0; chunkIndex < engine_t::kChunkLoadCount; ++chunkIndex)
        {
            engine_t::ChunkResources& chunk = ioEngine->chunks[chunkIndex];
            chunk.texture.reset(0u);
            glCreateTextures(GL_TEXTURE_3D, 1, chunk.texture.get());
            glTextureStorage3D(chunk.texture, 1, GL_R8,
                               1u << engine_t::kLog2ChunkSize,
                               1u << engine_t::kLog2ChunkSize,
                               1u << engine_t::kLog2ChunkSize);

            chunk.handle = glGetTextureSamplerHandleARB(chunk.texture, ioEngine->sampler);
        }

        ioEngine->chunk_texture.reset(0u);
        glCreateTextures(GL_TEXTURE_3D, 1, ioEngine->chunk_texture.get());
        glTextureStorage3D(ioEngine->chunk_texture, 1, GL_R8,
                           1u << engine_t::kLog2ChunkSize,
                           1u << engine_t::kLog2ChunkSize,
                           1u << engine_t::kLog2ChunkSize);

        ioEngine->chunk_handle = glGetTextureSamplerHandleARB(ioEngine->chunk_texture, ioEngine->sampler);
    }

    numtk::Vec3_t const eye_position = ioEngine->campos();
    numtk::Vec3i64_t const vs_eye_position = WS_to_VS(eye_position);

    numtk::Vec3i64_t const player_chunk_index = ComputeChunkIndex(vs_eye_position);
    ioEngine->campos_chunk_index = player_chunk_index;

    {
        using StdClock = std::chrono::high_resolution_clock;
        auto start = StdClock::now();

        numtk::Vec3i64_t const load_bounds_min =
            numtk::vec3i64_add(player_chunk_index, { -engine_t::kChunkLoadRadius, -engine_t::kChunkLoadRadius, -engine_t::kChunkLoadRadius });

        for (std::int64_t z = 0; z < engine_t::kChunkLoadExtent; ++z)
        {
            for (std::int64_t y = 0; y < engine_t::kChunkLoadExtent; ++y)
            {
                for (std::int64_t x = 0; x < engine_t::kChunkLoadExtent; ++x)
                {
                    numtk::Vec3i64_t const chunk_index =
                        numtk::vec3i64_add(load_bounds_min, { x, y, z });

                    engine_t::ChunkResources& chunk =
                        ioEngine->chunks[engine_t::kChunkLoadIndex({ x, y, z })];

                    chunk.base = ChunkBaseFromIndex(chunk_index);

                    if (!ioEngine->generated_chunks.count(chunk.base))
                    {
                        GenerateChunk(ioEngine->vdb, chunk.base);

                        if (kFragmentPipeline)
                            UploadChunk(chunk, ioEngine->vdb, ioEngine->staging_buffer);

                        ioEngine->generated_chunks.insert(chunk.base);
                    }
                }
            }
        }

        auto end = StdClock::now();
        float load_time =
            static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
        float chunk_average = load_time / (float)engine_t::kChunkLoadCount;
        std::cout << "Voxel generation " << load_time/1000.f << " s" << std::endl;
        std::cout << "  Chunk average " << chunk_average/1000.f << " s" << std::endl;
    } // VOXEL GENERATION

    if (kGeometryPipeline)
        RefreshRenderBuffer(ioEngine, { 0, 0, 0 });
}

void EngineRunFrame(engine_t* ioEngine, input_t const* iInput, float update_dt)
{
    //static constexpr float kFrameTime = 0.016f;
    const float kFrameTime = update_dt;

    if (false)
    {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(1ms);
    }

#if 0
    std::cout << "input time " << update_dt << std::endl;
    std::cout << iInput->key_down[' '] << " "
              << ((iInput->mod_down & fKeyMod::kShift) != 0) << std::endl
              << iInput->key_down['w'] << " "
              << iInput->key_down['a'] << " "
              << iInput->key_down['s'] << " "
              << iInput->key_down['d'] << std::endl;
#endif

    {

        if (iInput->key_down['0'])
        {
            ioEngine->player_position = { 0.f, 16.f, 0.f };
            ioEngine->player_velocity = { 0.f, -5.f, 0.f };
            ioEngine->player_on_ground = false;
        }

        if (!iInput->key_down['j'] && iInput->back_key_down['j']) // on release
        {
            ioEngine->noclip_mode = !ioEngine->noclip_mode;
            ioEngine->player_on_ground = false;
            std::cout << "Noclip mode " << (ioEngine->noclip_mode ? "enabled" : "disabled") << std::endl;
        }

        {
            static constexpr float kAngleSpeed = 0.5f;
            numtk::Vec3_t rotation_delta = numtk::vec3_float_mul(numtk::Vec3_t{
                -(float)iInput->mouse_delta[1],
                -(float)iInput->mouse_delta[0],
                0.f }
            , kAngleSpeed * (numtk::kPi / 180.f));

            ioEngine->camera_euler = numtk::vec3_add(ioEngine->camera_euler, rotation_delta);
            ioEngine->camera_euler[0] =
                std::max(std::min(ioEngine->camera_euler[0], numtk::kPi * .5f), -numtk::kPi * .5f);
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

            if (ioEngine->noclip_mode)
            {
                if ((iInput->mod_down & fKeyMod::kShift) != 0)
                    d = numtk::vec3_add(d, { 0.f, -1.f, 0.f });
                if (iInput->key_down[' '])
                    d = numtk::vec3_add(d, { 0.f, 1.f, 0.f });
            }

            // Apply input d
            if (numtk::vec3_dot(d, d) > 0.f)
            {
                d = numtk::vec3_normalise(d);

                static constexpr float kLineSpeed = 5.0f;
                numtk::Vec3_t dp = d;
                numtk::Mat4_t camrot = numtk::mat4_from_quat(ioEngine->camera_current);
                numtk::Vec3_t wsdp = numtk::vec3_from_vec4(numtk::mat4_vec4_mul(camrot, numtk::vec3_float_concat(dp, 0.f)));

                if (!ioEngine->noclip_mode)
                    wsdp[1] = 0.f;

                wsdp = numtk::vec3_normalise(wsdp);
                wsdp = numtk::vec3_float_mul(wsdp, kLineSpeed * kFrameTime);

                ioEngine->player_position = numtk::vec3_add(ioEngine->player_position, wsdp);
            }

            // Put voxel
#if 0
            if (iInput->key_down[' '])
            {
                quick_vdb::Position_t voxel{ ioEngine->look_at_voxel[0], ioEngine->look_at_voxel[1] + 1, ioEngine->look_at_voxel[2] };
                ioEngine->vdb.set(voxel);
                if (kGeometryPipeline)
                    RefreshRenderBuffer(ioEngine, { 0, 0, 0 });
            }
#endif
        }

        if (!ioEngine->noclip_mode)
        {
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
        }


        {
            ioEngine->last_voxel = WS_to_VS(ioEngine->player_position);
        }
    }

    {
        numtk::Mat4_t camrot = numtk::mat4_from_quat(ioEngine->camera_current);
        numtk::Vec4_t camforward = numtk::mat4_vec4_mul(camrot, numtk::Vec4_t{ 0.f, 0.f, -1.f, 0.f});
        numtk::Vec4_t camup = numtk::mat4_vec4_mul(camrot, numtk::Vec4_t{ 0.f, 1.f, 0.f, 0.f });

        numtk::Mat4_t viewmat = numtk::mat4_view(
            numtk::vec3_from_vec4(camforward),
            numtk::vec3_from_vec4(camup),
            ioEngine->campos()
        );

        numtk::Mat4_t invviewmat = numtk::mat4_viewinv(
            numtk::vec3_normalise(numtk::vec3_from_vec4(camforward)),
            numtk::vec3_normalise(numtk::vec3_from_vec4(camup)),
            numtk::Vec3_t{ 0.f, 0.f, 0.f }
        );

        float aspect_ratio = (float)iInput->screen_size[1] / (float)iInput->screen_size[0];
        float fov = numtk::kPi*0.5f;

        ioEngine->projection_matrix = numtk::mat4_mul(
            numtk::mat4_perspective(0.01f, 1000.f, fov, aspect_ratio),
            viewmat
        );

        ioEngine->projection_invmatrix = numtk::mat4_mul(
            invviewmat,
            numtk::mat4_perspectiveinv(0.01f, 1000.f, fov, aspect_ratio)
        );

#if 0
        {
            numtk::Vec3_t rd = numtk::vec3_from_vec4(numtk::mat4_vec4_mul(camrot, numtk::Vec4_t{ 0.f, 0.f, 1.f, 0.f }));

            numtk::Vec3_t p = ioEngine->campos();
            quick_vdb::Position_t vp = WS_to_VS(p);
            p = WS_to_VS_float(p);

#if 0
            std::cout << "last voxel " << ioEngine->last_voxel[0] << " " << ioEngine->last_voxel[1] << " " << ioEngine->last_voxel[2] << std::endl;
            std::cout << "ro " << p[0] << " " << p[1] << " " << p[2] << std::endl;
            std::cout << "rd " << rd[0] << " " << rd[1] << " " << rd[2] << std::endl;
#endif

            bool hit = false;
#if 0
            numtk::Vec3_t invrd = numtk::Vec3_t{ 1.f/rd[0], 1.f/rd[1], 1.f/rd[2] };
            for(;;)
            {
                if (numtk::vec3_norm(numtk::vec3_sub(VS_to_WS_float(p), ioEngine->campos())) > 5.f)
                    break;
                if (ioEngine->vdb.get(vp)) {
                    //std::cout << "Looking at " << vp[0] << " " << vp[1] << " " << vp[2] << std::endl;
                    hit = true; break; }

                //numtk::Vec3_t fract{ std::fmod(p[0], 1.f), std::fmod(p[1], 1.f), std::fmod(p[2], 1.f) };

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
#endif

            if (hit)
                ioEngine->look_at_voxel = vp;
            else
                ioEngine->look_at_voxel = { -1337, -1337, -1337 };
        }
#endif
    }

    if (kGeometryPipeline) // geometry shader pipeline
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

    if (kFragmentPipeline)
    {
        numtk::Vec3i64_t const campos_chunk_index = ComputeChunkIndex(WS_to_VS(ioEngine->campos()));

        if (campos_chunk_index != ioEngine->campos_chunk_index)
        {
            numtk::Vec3i64_t const delta = numtk::vec3i64_sub(campos_chunk_index, ioEngine->campos_chunk_index);

            numtk::Vec3i64_t const sign{ (delta[0] < 0) ? 0 : 1,
                (delta[1] < 0) ? 0 : 1,
                (delta[2] < 0) ? 0 : 1};

            numtk::Vec3i64_t const itcount{
                engine_t::kChunkLoadExtent - std::abs(delta[0]),
                engine_t::kChunkLoadExtent - std::abs(delta[1]),
                engine_t::kChunkLoadExtent - std::abs(delta[2])
            };

            std::uint32_t const keptChunkCount = itcount[0] * itcount[1] * itcount[2];
            std::vector<numtk::Vec3i64_t> loadQueue{};
            loadQueue.reserve(engine_t::kChunkLoadCount - keptChunkCount);

            for (std::int64_t z = 0; z < engine_t::kChunkLoadExtent; ++z)
            {
                std::int64_t const zIndex = sign[2] ? z : engine_t::kChunkLoadExtent-1 - z;
                for (std::int64_t y = 0; y < engine_t::kChunkLoadExtent; ++y)
                {
                    std::int64_t const yIndex = sign[1] ? y : engine_t::kChunkLoadExtent-1 - y;
                    for (std::int64_t x = 0; x < engine_t::kChunkLoadExtent; ++x)
                    {
                        std::int64_t const xIndex = sign[0] ? x : engine_t::kChunkLoadExtent-1 - x;
                        if (x < itcount[0] && y < itcount[1] && z < itcount[2])
                        {
                            // shift every cache entry in the direction opposite to delta
                            // missing chunks can be uploaded
                            numtk::Vec3i64_t const dst{ xIndex, yIndex, zIndex };
                            numtk::Vec3i64_t const src{
                                (xIndex + delta[0]),
                                (yIndex + delta[1]),
                                (zIndex + delta[2])
                            };

                            std::swap(ioEngine->chunks[engine_t::kChunkLoadIndex(dst)],
                                      ioEngine->chunks[engine_t::kChunkLoadIndex(src)]);
                        }
                        else
                            loadQueue.push_back({xIndex, yIndex, zIndex});
                    }
                }
            }

            for (numtk::Vec3i64_t const& chunkLoadOffset : loadQueue)
            {
                engine_t::ChunkResources& chunk =
                    ioEngine->chunks[engine_t::kChunkLoadIndex(chunkLoadOffset)];

                numtk::Vec3i64_t const load_region_begin = numtk::vec3i64_sub(
                    campos_chunk_index,
                    { engine_t::kChunkLoadRadius,
                      engine_t::kChunkLoadRadius,
                      engine_t::kChunkLoadRadius});

                numtk::Vec3i64_t const chunk_index = numtk::vec3i64_add(
                    load_region_begin, chunkLoadOffset);

                chunk.base = ChunkBaseFromIndex(chunk_index);

                if (!ioEngine->generated_chunks.count(chunk.base))
                {
                    GenerateChunk(ioEngine->vdb, chunk.base);
                    ioEngine->generated_chunks.insert(chunk.base);
                }

                if (kFragmentPipeline)
                    UploadChunk(chunk, ioEngine->vdb, ioEngine->staging_buffer);
            }

            ioEngine->campos_chunk_index = campos_chunk_index;
        }

        int const camchunk_index = engine_t::kChunkLoadIndex(
            numtk::vec3i64_add(
                numtk::vec3i64_sub(campos_chunk_index, ioEngine->campos_chunk_index),
                { engine_t::kChunkLoadRadius, engine_t::kChunkLoadRadius, engine_t::kChunkLoadRadius }
            ));

        engine_t::ChunkResources const& camchunk = ioEngine->chunks[camchunk_index];
        glMakeTextureHandleResidentARB(camchunk.handle);

        glViewport(0, 0, iInput->screen_size[0], iInput->screen_size[1]);

        glUseProgram(ioEngine->shader_program);
        {
            int const projmat_loc = glGetUniformLocation(ioEngine->shader_program, "iInvProj");
            if (projmat_loc >= 0)
                glUniformMatrix4fv(projmat_loc, 1, GL_FALSE, &ioEngine->projection_invmatrix[0]);

            int const extent_loc = glGetUniformLocation(ioEngine->shader_program, "iExtent");
            if (extent_loc >= 0)
                glUniform1f(extent_loc, engine_t::kVoxelScale);

            float resolution[2] { (float)iInput->screen_size[0], (float)iInput->screen_size[1] };
            int const resolution_loc = glGetUniformLocation(ioEngine->shader_program, "iResolution");
            if (resolution_loc >= 0)
                glUniform2fv(resolution_loc, 1, resolution);

            numtk::Vec3_t const eye_position = ioEngine->campos();
            numtk::Vec3i64_t const chunk_base =
                ioEngine->vdb.GetChildBase<0>(WS_to_VS(eye_position));

            numtk::Vec3_t const vs = WS_to_VS_float(eye_position);
            numtk::Vec3_t const fchunk_base{
                (float)chunk_base[0],
                (float)chunk_base[1],
                (float)chunk_base[2]
            };
            numtk::Vec3_t const chunklocal_vs = numtk::vec3_sub(vs, fchunk_base);

            numtk::Vec3_t const chunklocal_relative = numtk::vec3_float_mul(
                chunklocal_vs, 1.f / (float)engine_t::kChunkSize);

            int const chunkExtent_loc = glGetUniformLocation(ioEngine->shader_program, "iChunkExtent");
            if (chunkExtent_loc >= 0)
                glUniform1f(chunkExtent_loc, (float)engine_t::kChunkSize);

            int const chunkLocalCamPos_loc = glGetUniformLocation(ioEngine->shader_program, "iChunkLocalCamPos");
            if (chunkLocalCamPos_loc >= 0)
                glUniform3fv(chunkLocalCamPos_loc, 1, &chunklocal_relative[0]);

            int const chunk_loc = glGetUniformLocation(ioEngine->shader_program, "iChunk");
            if (chunk_loc >= 0)
                glUniformHandleui64ARB(chunk_loc, camchunk.handle);
        }

        glBindVertexArray(ioEngine->vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0u);

        glUseProgram(0u);
        glMakeTextureHandleNonResidentARB(camchunk.handle);
    }
}


}


oglbase::ShaderSources_t const rtvert{ "#version 430 core\n", R"__lstr__(

            const vec2 kTriVertices[] = vec2[3](
	            vec2(-1.0, 3.0), vec2(-1.0, -1.0), vec2(3.0, -1.0)
            );

            void main()
            {
	            gl_Position = vec4(kTriVertices[gl_VertexID], 0.0, 1.0);
            }

        )__lstr__"};

oglbase::ShaderSources_t const rtfrag{ "#version 430 core\n", R"__lstr__(

            #extension GL_ARB_bindless_texture : require

            vec3 raydir_frommat(mat4 perspective_inverse, vec2 clip_coord)
            {
                vec4 target = vec4(clip_coord, 1.0, 1.0);
                vec4 ray_direction = perspective_inverse * target;
                ray_direction = ray_direction / ray_direction.w;
                return normalize(ray_direction.xyz);
            }

            float maxc(vec3 v) {return max(max(v.x, v.y), v.z); }

            uniform mat4 iInvProj;
            uniform vec2 iResolution;

            uniform float iExtent;

            layout(bindless_sampler) uniform sampler3D iChunk;
            uniform float iChunkExtent;
            uniform vec3 iChunkLocalCamPos; // Normalized on chunk size

            layout(location = 0) out vec4 color;

            void main()
            {
                vec2 frag_coord = vec2(gl_FragCoord.xy);
	            vec2 clip_coord = ((frag_coord / iResolution) - 0.5) * 2.0;

                vec3 rd = raydir_frommat(iInvProj, clip_coord);
                vec3 ro = iChunkLocalCamPos;// - vec3(0.5);

#if 0
                ro = iChunkLocalCamPos - vec3(0.5);

                float winding = (maxc(abs(ro) * 2.0) < 1.0) ? -1.0 : 1.0;
                vec3 sgn = -sign(rd);
                vec3 d = (0.5 * winding * sgn - ro) / rd;

            #define TEST(U, V, W) \
                (d.U >= 0.0) && all(lessThan(abs(vec2(ro.V, ro.W) + vec2(rd.V, rd.W)*d.U), vec2(0.5)))

            bvec3 test = bvec3(
                TEST(x, y, z),
                TEST(y, z, x),
                TEST(z, x, y));

            #undef TEST


            sgn = test.x ? vec3(sgn.x,0,0) : (test.y ? vec3(0,sgn.y,0) : vec3(0, 0,test.z ? sgn.z : 0));

            float distance = (sgn.x != 0) ? d.x : ((sgn.y != 0) ? d.y : d.z);
            vec3 normal = sgn;
            bool hit = (sgn.x != 0) || (sgn.y != 0) || (sgn.z != 0);

                // bounds intersection
                color = (vec4(rd, 1.0));

                if (hit)
                {
                    color = vec4(normal * 0.5 + vec3(0.5), 1.0);
                    vec3 puvw = (ro + rd * distance * 1.0001) * 0.5 + vec3(0.5);
                    vec3 pvoxel = puvw * iChunkExtent;

                    color.xyz = vec3(distance*0.5);//texture(iChunk, puvw).xxx;
                }
#else

                vec3 stepSign = sign(rd);
                vec3 vsro = ro * iChunkExtent;
                vec3 p = floor(vsro);

                vec3 manh = stepSign * (vec3(0.5) - fract(vsro)) + 0.5;
                //vec3 manh = mix(fract(vsro), 1.0 - fract(vsro), stepSign * 0.5 + 0.5);
                vec3 compv = (stepSign * 0.5 + vec3(0.5));

                vec3 tDelta = 1.f / abs(rd);
                vec3 tMax = manh * tDelta;
                vec3 uvwstep = stepSign / iChunkExtent;
                vec3 puvw = (p + vec3(0.5))  / iChunkExtent;
                vec3 start = puvw;
                vec3 temp = puvw;

                //color.xyz = tMax;

#if 1
                float accum = 0.f;
                float t = 0.f;
                while( (puvw.x * stepSign.x < compv.x)
                    && (puvw.y * stepSign.y < compv.y)
                    && (puvw.z * stepSign.z < compv.z))
                {

temp = puvw;
accum += texture(iChunk, puvw).x;// / 32.f;
if (accum.x >= 1.0) break;

if (tMax.x < tMax.y)
{
    if (tMax.x < tMax.z)
    {
        puvw.x += uvwstep.x;
        tMax.x += tDelta.x;
        t += tDelta.x;
    }
    else
    {
        puvw.z += uvwstep.z;
        tMax.z += tDelta.z;
        t += tDelta.z;
    }
}

else
{
    if (tMax.y < tMax.z)
    {
        puvw.y += uvwstep.y;
        tMax.y += tDelta.y;
        t += tDelta.y;
    }
    else
    {
        puvw.z += uvwstep.z;
        tMax.z += tDelta.z;
        t += tDelta.z;
    }
}

                }

                color.xyz = vec3(puvw * (0.1 / distance(puvw,start)));
                    //distance(puvw, start));//puvw * 0.5;// + vec3(0.5);
#endif

#endif
            }

        )__lstr__" };
