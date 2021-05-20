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
    bool lock_load_area = false;
    bool player_on_ground = false;
    numtk::Vec3i64_t last_voxel{};
    numtk::Vec3i64_t look_at_voxel{ -1337, -1337, -1337 };

    static constexpr unsigned kLog2ChunkSize = 6u;
    static constexpr unsigned kChunkSize = 1u << kLog2ChunkSize;
    static constexpr int kChunkLoadRadius = 2;
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

void GenerateChunk(engine_t::VDB_t& vdb, numtk::Vec3i64_t const& chunk_base)
{
    Timer<Verbose> timer0("GenerateChunk");

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
                    return ((voxel_world[0] & 3u)
                            + (voxel_world[1] & 3u)
                            + (voxel_world[2] & 3u)) < 2u;
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
    Timer<Verbose> timer0("UploadChunk");

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
    if (!ioEngine->lock_load_area)
        ioEngine->campos_chunk_index = ComputeChunkIndex(vs_eye_position);

    {
        using StdClock = std::chrono::high_resolution_clock;
        auto start = StdClock::now();

        numtk::Vec3i64_t const load_bounds_min =
            numtk::vec3i64_add(ioEngine->campos_chunk_index, { -engine_t::kChunkLoadRadius, -engine_t::kChunkLoadRadius, -engine_t::kChunkLoadRadius });

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

        if (!iInput->key_down['k'] && iInput->back_key_down['k']) // on release
        {
            ioEngine->lock_load_area = !ioEngine->lock_load_area;
            std::cout << "Lock area " << (ioEngine->lock_load_area ? "enabled" : "disabled") << std::endl;
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

                static constexpr float kLineSpeed = 50.0f;
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

    if (kFragmentPipeline)
    {
        numtk::Vec3i64_t const campos_chunk_index = ComputeChunkIndex(WS_to_VS(ioEngine->campos()));

        if (!ioEngine->lock_load_area && campos_chunk_index != ioEngine->campos_chunk_index)
        {
            Timer<Info> timer0("ChunkReloc");

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

            int const chunkExtent_loc = glGetUniformLocation(ioEngine->shader_program, "iChunkExtent");
            if (chunkExtent_loc >= 0)
                glUniform1f(chunkExtent_loc, (float)engine_t::kChunkSize);
        }

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);

        static GLfloat const clear_color[]{ 0.5f, 0.5f, 0.5f, 1.f };
        glClearBufferfv(GL_COLOR, 0, clear_color);
        static GLfloat const clear_depth = 1.f;
        glClearBufferfv(GL_DEPTH, 0, &clear_depth);

        for (int chunkIndex = 0u; chunkIndex < engine_t::kChunkLoadCount; ++chunkIndex)
        {
            engine_t::ChunkResources const& chunk = ioEngine->chunks[chunkIndex];
            glMakeTextureHandleResidentARB(chunk.handle);

            numtk::Vec3_t const eye_position = ioEngine->campos();

            numtk::Vec3_t const vs = WS_to_VS_float(eye_position);
            numtk::Vec3_t const fchunk_base{
                (float)chunk.base[0],
                (float)chunk.base[1],
                (float)chunk.base[2]
            };
            numtk::Vec3_t const chunklocal_vs = numtk::vec3_sub(vs, fchunk_base);

            numtk::Vec3_t const chunklocal_relative = numtk::vec3_float_mul(
                chunklocal_vs, 1.f / (float)engine_t::kChunkSize);

            int const chunkLocalCamPos_loc = glGetUniformLocation(ioEngine->shader_program, "iChunkLocalCamPos");
            if (chunkLocalCamPos_loc >= 0)
                glUniform3fv(chunkLocalCamPos_loc, 1, &chunklocal_relative[0]);

            int const chunk_loc = glGetUniformLocation(ioEngine->shader_program, "iChunk");
            if (chunk_loc >= 0)
                glUniformHandleui64ARB(chunk_loc, chunk.handle);

            glBindVertexArray(ioEngine->vao);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glBindVertexArray(0u);

            glMakeTextureHandleNonResidentARB(chunk.handle);
        }

        glUseProgram(0u);
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

oglbase::ShaderSources_t const rtfrag{
    #include "voxeltraversal.frag.glsl"
};
