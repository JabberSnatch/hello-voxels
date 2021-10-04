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
#include "rng.h"

constexpr bool kFragmentPipeline = true;

extern oglbase::ShaderSources_t const rtvert;
extern oglbase::ShaderSources_t const rtfrag;

extern oglbase::ShaderSources_t const layeredgeom;

extern oglbase::ShaderSources_t const skytrfrag;
extern oglbase::ShaderSources_t const skydirfrag;
extern oglbase::ShaderSources_t const skysscatfrag;
extern oglbase::ShaderSources_t const skyscatdfrag;
extern oglbase::ShaderSources_t const skyiirfrag;
extern oglbase::ShaderSources_t const skymscatfrag;

struct atmosphere_t
{
    numtk::Vec3_t sun_irradiance;
    float sun_angular_radius;
    numtk::Vec2_t bounds;
    float mus_min;
    float padding0;
    numtk::Vec4_t rscat;
    numtk::Vec4_t mext;
    numtk::Vec3_t mscat;
    float padding1;
    float odensity[8];
    numtk::Vec4_t oext;
};

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

    // bool render_data_clean = false;
    // std::vector<numtk::Vec3_t> points{};

    oglbase::BufferPtr vbo{};
    oglbase::VAOPtr vao{};
    oglbase::ProgramPtr shader_program{};

    oglbase::BufferPtr staging_buffer{};
    oglbase::SamplerPtr sampler{};

    numtk::SH2nd_t sh_data[3];

    oglbase::BufferPtr atmosBuffer{};
    oglbase::TexturePtr trtex{};
    GLuint64 trhandle{};
    oglbase::TexturePtr iirtex{};
    GLuint64 iirhandle{};
    oglbase::TexturePtr sctex{};
    GLuint64 schandle{};

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

void ComputeClearSky();

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

    float heightmap[engine_t::kChunkSize * engine_t::kChunkSize];
    {

        for (std::int64_t vz = 0; vz < engine_t::kChunkSize; ++vz)
        {
            for (std::int64_t vx = 0; vx < engine_t::kChunkSize; ++vx)
            {
                numtk::Vec3i64_t const voxel_index{ vx, 0, vz };
                numtk::Vec3i64_t const voxel_world = numtk::vec3i64_add(chunk_voxel_base, voxel_index);

                numtk::Vec3_t fvoxel_world = VS_to_WS(voxel_world);
                fvoxel_world =
                    numtk::vec3_add(fvoxel_world,
                                    numtk::vec3_constant(0.5f * engine_t::kVoxelScale));

                numtk::Vec3_t p{ fvoxel_world[0] * 0.05f, 0.f, fvoxel_world[2] * 0.05f };

                heightmap[vz + vx*engine_t::kChunkSize] = rng::noise3(&p[0]) * 20.f;
            }
        }
    }

    for (std::int64_t vz = 0; vz < engine_t::kChunkSize; ++vz)
    {
        for (std::int64_t vy = 0; vy < engine_t::kChunkSize; ++vy)
        {
            for (std::int64_t vx = 0; vx < engine_t::kChunkSize; ++vx)
            {
                numtk::Vec3i64_t voxel_index{ vx, vy, vz };
                numtk::Vec3i64_t voxel_world = numtk::vec3i64_add(chunk_voxel_base, voxel_index);

                bool set_voxel = [&heightmap, vx, vz](numtk::Vec3i64_t const& voxel_world)
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

                    static const float freq = 10.f;
                    static const float hfreq = freq*0.5f;
                    numtk::Vec3_t repeat{
                        std::copysign(
                            std::fmod(std::abs(fvoxel_world[0]) + hfreq, freq) - hfreq,
                            fvoxel_world[0]
                        ),
                        std::copysign(
                            std::fmod(std::abs(fvoxel_world[1]) + hfreq, freq) - hfreq,
                            fvoxel_world[1]
                        ),
                        std::copysign(
                            std::fmod(std::abs(fvoxel_world[2]) + hfreq, freq) - hfreq,
                            fvoxel_world[2]
                        )
                    };

#if 0
                    float distance = numtk::vec3_norm(repeat);
                    //return distance * (std::cos(distance * 1.618033f) + 1.f) < 2.f;
                    return distance < 2.f * (std::fmod(std::cos(distance * 1.618033f), 1.f) + 1.f);
#endif

                    float radius = numtk::vec3_norm({ fvoxel_world[0], 0.f, fvoxel_world[2] });
                    float otherradius = numtk::vec3_norm({ fvoxel_world[0] - 2.f, 0.f, fvoxel_world[2]+10.f });

                    bool floor = fvoxel_world[1] <= (heightmap[vz + vx*engine_t::kChunkSize]);// + (std::cos(radius*5.f) * 0.5f) + std::cos(otherradius*0.2f) * 4.f);
                    bool ceil = fvoxel_world[1] >= -heightmap[vz + vx*engine_t::kChunkSize] + 100.f;// + (std::cos(radius*5.f) * 0.5f) + std::cos(otherradius*0.2f) * 4.f;

                    bool column = false;
                    {
                        static const float freq = 66.f;
                        static const float hfreq = freq*0.5f;
                        numtk::Vec3_t repeat{
                            std::copysign(
                                std::fmod(std::abs(fvoxel_world[0] + hfreq) + hfreq, freq) - hfreq,
                                fvoxel_world[0]
                            ),
                            std::copysign(
                                std::fmod(std::abs(fvoxel_world[1] + hfreq) + hfreq, freq) - hfreq,
                                fvoxel_world[1]
                            ),
                            std::copysign(
                                std::fmod(std::abs(fvoxel_world[2] + hfreq) + hfreq, freq) - hfreq,
                                fvoxel_world[2]
                            )
                        };

                        float theta = std::atan2(repeat[2], repeat[0]);
                        float distance = std::sqrt(repeat[0] * repeat[0] + repeat[2] * repeat[2]);
                        float height = std::max(0.f, std::min(100.f, fvoxel_world[1])) / 100.f;

                        const float radius = 7.f;
                        const float baseheight = 0.075f;

                        if (height < baseheight || height > 1.f - baseheight)
                        {
                            column = distance < 9.f + std::sin(theta * 50.f) * 0.4f;
                        }
                        else
                        {
                            column = distance < 7.f + std::sin(height * numtk::kPi * 3.f) * 0.2f;
                        }
                    }

                    return floor || ceil || column;

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

void ComputeClearSky()
{
    // transmittance
    // direct irradiance
    // rayleigh + mie single scattering
    // rayleigh + mie multiple scattering
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

    {
        // Precomputed Atmospheric Scattering (Bruneton & Neyret, 2008)

        constexpr int kTrTexWidth = 256;
        constexpr int kTrTexHeight = 64;

        constexpr int kIrTexWidth = 64;
        constexpr int kIrTexHeight = 16;

        constexpr int kScRSize = 32;
        constexpr int kScMuVSize = 128;
        constexpr int kScMuSSize = 32;
        constexpr int kScNuSize = 8;

        constexpr int kScTexWidth = kScNuSize * kScMuSSize;
        constexpr int kScTexHeight = kScMuVSize;
        constexpr int kScTexDepth = kScRSize;

        oglbase::SamplerPtr sampler{};
        glGenSamplers(1, sampler.get());
        glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        oglbase::TexturePtr& trtex = ioEngine->trtex;
        glCreateTextures(GL_TEXTURE_2D, 1, trtex.get());
        glBindTexture(GL_TEXTURE_2D, trtex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, kTrTexWidth, kTrTexHeight, 0, GL_RGB, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        oglbase::TexturePtr dirtex{};
        glCreateTextures(GL_TEXTURE_2D, 1, dirtex.get());
        glBindTexture(GL_TEXTURE_2D, dirtex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, kIrTexWidth, kIrTexHeight, 0, GL_RGB, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        oglbase::TexturePtr& iirtex = ioEngine->iirtex;
        glCreateTextures(GL_TEXTURE_2D, 1, iirtex.get());
        glBindTexture(GL_TEXTURE_2D, iirtex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, kIrTexWidth, kIrTexHeight, 0, GL_RGB, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        oglbase::TexturePtr& sctex = ioEngine->sctex;
        glCreateTextures(GL_TEXTURE_3D, 1, sctex.get());
        glBindTexture(GL_TEXTURE_3D, sctex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, kScTexWidth, kScTexHeight, kScTexDepth, 0, GL_RGBA, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_3D, 0);

        oglbase::TexturePtr drsctex{};
        glCreateTextures(GL_TEXTURE_3D, 1, drsctex.get());
        glBindTexture(GL_TEXTURE_3D, drsctex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, kScTexWidth, kScTexHeight, kScTexDepth, 0, GL_RGB, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_3D, 0);

        oglbase::TexturePtr dmsctex{};
        glCreateTextures(GL_TEXTURE_3D, 1, dmsctex.get());
        glBindTexture(GL_TEXTURE_3D, dmsctex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, kScTexWidth, kScTexHeight, kScTexDepth, 0, GL_RGB, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_3D, 0);

        oglbase::TexturePtr dscdtex{};
        glCreateTextures(GL_TEXTURE_3D, 1, dscdtex.get());
        glBindTexture(GL_TEXTURE_3D, dscdtex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, kScTexWidth, kScTexHeight, kScTexDepth, 0, GL_RGB, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_3D, 0);

        oglbase::TexturePtr msctex{}; // Can be set to reuse drsctex rather than get its own alloc.
        glCreateTextures(GL_TEXTURE_3D, 1, msctex.get());
        glBindTexture(GL_TEXTURE_3D, msctex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, kScTexWidth, kScTexHeight, kScTexDepth, 0, GL_RGB, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_3D, 0);

        oglbase::FBOPtr fbo{};
        glGenFramebuffers(1, fbo.get());

        std::string log{};
        oglbase::ShaderPtr vbin = oglbase::CompileShader(GL_VERTEX_SHADER, rtvert, &log);
        std::cout << "Vshader " << log << std::endl;
        oglbase::ShaderPtr gbin = oglbase::CompileShader(GL_GEOMETRY_SHADER, layeredgeom, &log);
        std::cout << "Gshader " << log << std::endl;

        oglbase::ShaderPtr skytrbin = oglbase::CompileShader(GL_FRAGMENT_SHADER, skytrfrag, &log);
        std::cout << "skytr " << log << std::endl;
        oglbase::ProgramPtr skytrprog = oglbase::LinkProgram({ skytrbin, vbin }, &log);
        std::cout << "skytrprog " << log << std::endl;

        oglbase::ShaderPtr skydirbin = oglbase::CompileShader(GL_FRAGMENT_SHADER, skydirfrag, &log);
        std::cout << "skydir " << log << std::endl;
        oglbase::ProgramPtr skydirprog = oglbase::LinkProgram({ skydirbin, vbin }, &log);
        std::cout << "skydirprog " << log << std::endl;

        oglbase::ShaderPtr skysscatbin = oglbase::CompileShader(GL_FRAGMENT_SHADER, skysscatfrag, &log);
        std::cout << "skysscat " << log << std::endl;
        oglbase::ProgramPtr skysscatprog = oglbase::LinkProgram({ skysscatbin, vbin, gbin }, &log);
        std::cout << "skysscatprog " << log << std::endl;

        oglbase::ShaderPtr skyscatdbin = oglbase::CompileShader(GL_FRAGMENT_SHADER, skyscatdfrag, &log);
        std::cout << "skyscatd " << log << std::endl;
        oglbase::ProgramPtr skyscatdprog = oglbase::LinkProgram({ skyscatdbin, vbin, gbin }, &log);
        std::cout << "skyscatdprog " << log << std::endl;

        oglbase::ShaderPtr skyiirbin = oglbase::CompileShader(GL_FRAGMENT_SHADER, skyiirfrag, &log);
        std::cout << "skyiir " << log << std::endl;
        oglbase::ProgramPtr skyiirprog = oglbase::LinkProgram({ skyiirbin, vbin }, &log);
        std::cout << "skyiirprog " << log << std::endl;

        oglbase::ShaderPtr skymscatbin = oglbase::CompileShader(GL_FRAGMENT_SHADER, skymscatfrag, &log);
        std::cout << "skymscat " << log << std::endl;
        oglbase::ProgramPtr skymscatprog = oglbase::LinkProgram({ skymscatbin, vbin, gbin }, &log);
        std::cout << "skymscatprog " << log << std::endl;

        oglbase::VAOPtr vao{};
        glGenVertexArrays(1, vao.get());

        numtk::Vec3_t lambda{ 610.e-3f, 550.e-3f, 450.e-3f };

        numtk::Vec3_t la0{ std::pow(lambda[0], -4.f),
            std::pow(lambda[1], -4.f),
            std::pow(lambda[2], -4.f)
        };
        numtk::Vec3_t lar = numtk::vec3_float_mul(la0, 1.24062e-6f);
        numtk::Vec4_t rscat{ lar[0], lar[1], lar[2], -1.f / 8.e3f };

        //kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha);
        // (kMieAnstromAlpha set to 0.0)
        float mie = 5.328e-3f / 1.2e3f;
        numtk::Vec4_t mext{ mie, mie, mie, -1.f / 1.2e3f };
        std::cout << "mie ext " << mext[0] << " " << mext[1] << " " << mext[2] << std::endl;

        numtk::Vec3_t mscat{ .9f * mext[0], .9f * mext[1], .9f * mext[2] };

        float ozone = 300.f * 2.687e20f / 1.5e4f;
        numtk::Vec4_t oext{ 4.228e-26f * ozone, 4.305e-25f * ozone, 2.316e-26f * ozone, 2.5e4f };

        atmosphere_t atmosphere{
            { 1.f, 1.f, 1.f },//numtk::Vec3_t sun_irradiance;
            .00935f * .5f,//float sun_angular_radius;
            { 6.36e6f, 6.42e6f },//numtk::Vec2_t bounds;
            std::cos(102.f * (numtk::kPi / 180.f)),
            0.f,
            rscat, mext, mscat,
            0.f,
            { 1.f / 1.5e4f, -2.f / 3.f, 0.f, 0.f, -1.f / 1.5e4f, 8.f / 3.f, 0.f, 0.f },//float odensity[4];
            oext
        };

        oglbase::BufferPtr& atmosphereBuffer = ioEngine->atmosBuffer;
        glCreateBuffers(1, atmosphereBuffer.get());

        glBindBuffer(GL_UNIFORM_BUFFER, atmosphereBuffer);
        glBufferData(GL_UNIFORM_BUFFER,
                     sizeof(atmosphere_t),
                     &atmosphere,
                     GL_STATIC_DRAW);

        glBindBuffer(GL_UNIFORM_BUFFER, 0u);

        {
            numtk::Vec2_t resolution{(float)kTrTexWidth, (float)kTrTexHeight};
            oglbase::BufferPtr viewportBuffer{};
            glCreateBuffers(1, viewportBuffer.get());
            glBindBuffer(GL_UNIFORM_BUFFER, viewportBuffer);
            glBufferData(GL_UNIFORM_BUFFER,
                         2 * sizeof(float),
                         &resolution[0],
                         GL_STATIC_DRAW);

            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, trtex, 0);

            glDrawBuffer(GL_COLOR_ATTACHMENT0);

            GLenum fboCheck = glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
            if (fboCheck != GL_FRAMEBUFFER_COMPLETE)
                std::cout << "Framebuffer status error : " << fboCheck << std::endl;

            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);

            glViewport(0, 0, kTrTexWidth, kTrTexHeight);

            glUseProgram(skytrprog);

            //GLuint index = glGetUniformBlockIndex(skytrprog, "ViewportBlock");
            //glUniformBlockBinding(skytrprog, index, 0u);
            glBindBufferBase(GL_UNIFORM_BUFFER, 0u, viewportBuffer);

            //index = glGetUniformBlockIndex(skytrprog, "AtmosphereBlock");
            //glUniformBlockBinding(skytrprog, index, 1u);
            glBindBufferBase(GL_UNIFORM_BUFFER, 1u, atmosphereBuffer);

            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glBindVertexArray(0u);
        }

        {
            numtk::Vec2_t resolution{(float)kIrTexWidth, (float)kIrTexHeight};
            oglbase::BufferPtr viewportBuffer{};
            glCreateBuffers(1, viewportBuffer.get());
            glBindBuffer(GL_UNIFORM_BUFFER, viewportBuffer);
            glBufferData(GL_UNIFORM_BUFFER,
                         2 * sizeof(float),
                         &resolution[0],
                         GL_STATIC_DRAW);

            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dirtex, 0);

            glDrawBuffer(GL_COLOR_ATTACHMENT0);

            GLenum fboCheck = glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
            if (fboCheck != GL_FRAMEBUFFER_COMPLETE)
                std::cout << "Framebuffer status error : " << fboCheck << std::endl;

            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);

            glViewport(0, 0, kIrTexWidth, kIrTexHeight);

            glUseProgram(skydirprog);

            glBindBufferBase(GL_UNIFORM_BUFFER, 0u, viewportBuffer);
            glBindBufferBase(GL_UNIFORM_BUFFER, 1u, atmosphereBuffer);

            GLuint64 handle = glGetTextureSamplerHandleARB(trtex, sampler);
            glMakeTextureHandleResidentARB(handle);

            glUniformHandleui64ARB(glGetUniformLocation(skydirprog, "trtex"),
                                   handle);

            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glBindVertexArray(0u);

            glMakeTextureHandleNonResidentARB(handle);
        }

        {
            numtk::Vec4_t const resolution{
                (float)kScNuSize, (float)kScMuSSize, (float)kScMuVSize, (float)kScRSize
            };

            oglbase::BufferPtr viewportBuffer{};
            glCreateBuffers(1, viewportBuffer.get());
            glBindBuffer(GL_UNIFORM_BUFFER, viewportBuffer);
            glBufferData(GL_UNIFORM_BUFFER,
                         4 * sizeof(float),
                         &resolution[0],
                         GL_STATIC_DRAW);

            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, drsctex, 0);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, dmsctex, 0);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, sctex, 0);

            static GLuint const draw_buffers[3] {
                GL_COLOR_ATTACHMENT0,
                GL_COLOR_ATTACHMENT1,
                GL_COLOR_ATTACHMENT2,
            };

            glDrawBuffers(3, draw_buffers);

            GLenum fboCheck = glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
            if (fboCheck != GL_FRAMEBUFFER_COMPLETE)
                std::cout << "Framebuffer status error : " << fboCheck << std::endl;

            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);

            glViewport(0, 0, kScTexWidth, kScTexHeight);

            glUseProgram(skysscatprog);

            glBindBufferBase(GL_UNIFORM_BUFFER, 0u, viewportBuffer);
            glBindBufferBase(GL_UNIFORM_BUFFER, 1u, atmosphereBuffer);

            GLuint64 handle = glGetTextureSamplerHandleARB(trtex, sampler);
            glMakeTextureHandleResidentARB(handle);

            glUniformHandleui64ARB(glGetUniformLocation(skysscatprog, "trtex"),
                                   handle);

            glBindVertexArray(vao);
            for (int i = 0; i < kScTexDepth; ++i)
            {
                glUniform1f(glGetUniformLocation(skysscatprog, "layer"),
                            (float)i);

                glDrawArrays(GL_TRIANGLES, 0, 3);
            }
            glBindVertexArray(0u);

            glMakeTextureHandleNonResidentARB(handle);

            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_NONE, 0);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_NONE, 0);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_NONE, 0);
        }

        for (unsigned order = 2; order <= 4; ++order)
        {
            { // scattering density
                numtk::Vec4_t const resolution{
                    (float)kScNuSize, (float)kScMuSSize, (float)kScMuVSize, (float)kScRSize
                };

                oglbase::BufferPtr viewportBuffer{};
                glCreateBuffers(1, viewportBuffer.get());
                glBindBuffer(GL_UNIFORM_BUFFER, viewportBuffer);
                glBufferData(GL_UNIFORM_BUFFER,
                             4 * sizeof(float),
                             &resolution[0],
                             GL_STATIC_DRAW);

                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, dscdtex, 0);
                glDrawBuffer(GL_COLOR_ATTACHMENT0);

                GLenum fboCheck = glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
                if (fboCheck != GL_FRAMEBUFFER_COMPLETE)
                    std::cout << "Framebuffer status error : " << fboCheck << std::endl;

                glDisable(GL_DEPTH_TEST);
                glDisable(GL_BLEND);

                glViewport(0, 0, kScTexWidth, kScTexHeight);

                glUseProgram(skyscatdprog);

                glBindBufferBase(GL_UNIFORM_BUFFER, 0u, viewportBuffer);
                glBindBufferBase(GL_UNIFORM_BUFFER, 1u, atmosphereBuffer);

                GLuint64 trhandle = glGetTextureSamplerHandleARB(trtex, sampler);
                glMakeTextureHandleResidentARB(trhandle);
                glUniformHandleui64ARB(glGetUniformLocation(skyscatdprog, "trtex"),
                                       trhandle);

                GLuint64 drschandle = glGetTextureSamplerHandleARB(drsctex, sampler);
                GLuint64 dmschandle = glGetTextureSamplerHandleARB(dmsctex, sampler);
                GLuint64 mschandle = glGetTextureSamplerHandleARB(msctex, sampler);

                glMakeTextureHandleResidentARB(drschandle);
                glUniformHandleui64ARB(glGetUniformLocation(skyscatdprog, "drsctex"),
                                       drschandle);

                glMakeTextureHandleResidentARB(dmschandle);
                glUniformHandleui64ARB(glGetUniformLocation(skyscatdprog, "dmsctex"),
                                       dmschandle);

                glMakeTextureHandleResidentARB(mschandle);
                glUniformHandleui64ARB(glGetUniformLocation(skyscatdprog, "msctex"),
                                       mschandle);

                GLuint64 dirhandle = glGetTextureSamplerHandleARB(dirtex, sampler);
                glMakeTextureHandleResidentARB(dirhandle);
                glUniformHandleui64ARB(glGetUniformLocation(skyscatdprog, "dirtex"),
                                       dirhandle);

                glUniform1f(glGetUniformLocation(skyscatdprog, "scatorder"),
                            (float)order);

                glBindVertexArray(vao);
                for (int i = 0; i < kScTexDepth; ++i)
                {
                    glUniform1f(glGetUniformLocation(skyscatdprog, "layer"),
                                (float)i);

                    glDrawArrays(GL_TRIANGLES, 0, 3);
                }
                glBindVertexArray(0u);

                glMakeTextureHandleNonResidentARB(trhandle);

                glMakeTextureHandleNonResidentARB(drschandle);
                glMakeTextureHandleNonResidentARB(dmschandle);
                glMakeTextureHandleNonResidentARB(mschandle);

                glMakeTextureHandleNonResidentARB(dirhandle);
            }

            { // indirect irradiance
                numtk::Vec2_t const resolution{(float)kIrTexWidth, (float)kIrTexHeight};
                oglbase::BufferPtr viewportBuffer{};
                glCreateBuffers(1, viewportBuffer.get());
                glBindBuffer(GL_UNIFORM_BUFFER, viewportBuffer);
                glBufferData(GL_UNIFORM_BUFFER,
                             2 * sizeof(float),
                             &resolution[0],
                             GL_STATIC_DRAW);

                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, dirtex, 0);
                glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, iirtex, 0);

                static GLuint const draw_buffers[2] {
                    GL_COLOR_ATTACHMENT0,
                    GL_COLOR_ATTACHMENT1,
                };

                glDrawBuffers(2, draw_buffers);

                GLenum fboCheck = glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
                if (fboCheck != GL_FRAMEBUFFER_COMPLETE)
                    std::cout << "Framebuffer status error : " << fboCheck << std::endl;

                glBlendEquation(GL_FUNC_ADD);
                glBlendFunc(GL_ONE, GL_ONE);

                glDisablei(GL_BLEND, 0);
                glEnablei(GL_BLEND, 1);

                glDisable(GL_DEPTH_TEST);

                glViewport(0, 0, kIrTexWidth, kIrTexHeight);

                glUseProgram(skyiirprog);

                glBindBufferBase(GL_UNIFORM_BUFFER, 0u, viewportBuffer);
                glBindBufferBase(GL_UNIFORM_BUFFER, 1u, atmosphereBuffer);

                GLuint64 drschandle = glGetTextureSamplerHandleARB(drsctex, sampler);
                GLuint64 dmschandle = glGetTextureSamplerHandleARB(dmsctex, sampler);
                GLuint64 mschandle = glGetTextureSamplerHandleARB(msctex, sampler);

                glMakeTextureHandleResidentARB(drschandle);
                glUniformHandleui64ARB(glGetUniformLocation(skyiirprog, "drsctex"),
                                       drschandle);

                glMakeTextureHandleResidentARB(dmschandle);
                glUniformHandleui64ARB(glGetUniformLocation(skyiirprog, "dmsctex"),
                                       dmschandle);

                glMakeTextureHandleResidentARB(mschandle);
                glUniformHandleui64ARB(glGetUniformLocation(skyiirprog, "msctex"),
                                       mschandle);

                glUniform1f(glGetUniformLocation(skyiirprog, "scatorder"),
                            (float)order);

                glUniform1f(glGetUniformLocation(skyiirprog, "nu_tex_size"),
                            (float)kScNuSize);

                glBindVertexArray(vao);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                glBindVertexArray(0u);

                glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_NONE, 0);
                glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_NONE, 0);

                glMakeTextureHandleNonResidentARB(drschandle);
                glMakeTextureHandleNonResidentARB(dmschandle);
                glMakeTextureHandleNonResidentARB(mschandle);
            }

            { // multiple scattering
                numtk::Vec4_t const resolution{
                    (float)kScNuSize, (float)kScMuSSize, (float)kScMuVSize, (float)kScRSize
                };

                oglbase::BufferPtr viewportBuffer{};
                glCreateBuffers(1, viewportBuffer.get());
                glBindBuffer(GL_UNIFORM_BUFFER, viewportBuffer);
                glBufferData(GL_UNIFORM_BUFFER,
                             4 * sizeof(float),
                             &resolution[0],
                             GL_STATIC_DRAW);

                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, msctex, 0);
                glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, sctex, 0);

                static GLuint const draw_buffers[2] {
                    GL_COLOR_ATTACHMENT0,
                    GL_COLOR_ATTACHMENT1,
                };

                glDrawBuffers(2, draw_buffers);

                GLenum fboCheck = glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
                if (fboCheck != GL_FRAMEBUFFER_COMPLETE)
                    std::cout << "Framebuffer status error : " << fboCheck << std::endl;

                glBlendEquation(GL_FUNC_ADD);
                glBlendFunc(GL_ONE, GL_ONE);

                glDisablei(GL_BLEND, 0);
                glEnablei(GL_BLEND, 1);

                glDisable(GL_DEPTH_TEST);

                glViewport(0, 0, kScTexWidth, kScTexHeight);

                glUseProgram(skymscatprog);

                glBindBufferBase(GL_UNIFORM_BUFFER, 0u, viewportBuffer);
                glBindBufferBase(GL_UNIFORM_BUFFER, 1u, atmosphereBuffer);

                GLuint64 trhandle = glGetTextureSamplerHandleARB(trtex, sampler);
                glMakeTextureHandleResidentARB(trhandle);
                glUniformHandleui64ARB(glGetUniformLocation(skymscatprog, "trtex"),
                                       trhandle);

                GLuint64 dscdhandle = glGetTextureSamplerHandleARB(dscdtex, sampler);
                glMakeTextureHandleResidentARB(dscdhandle);
                glUniformHandleui64ARB(glGetUniformLocation(skymscatprog, "dscdtex"),
                                       dscdhandle);

                glBindVertexArray(vao);
                for (int i = 0; i < kScTexDepth; ++i)
                {
                    glUniform1f(glGetUniformLocation(skymscatprog, "layer"),
                                (float)i);

                    glDrawArrays(GL_TRIANGLES, 0, 3);
                }
                glBindVertexArray(0u);

                glMakeTextureHandleNonResidentARB(trhandle);
                glMakeTextureHandleNonResidentARB(dscdhandle);

                glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_NONE, 0);
                glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_NONE, 0);
            }
        }

        glUseProgram(0u);
        glBindFramebuffer(GL_FRAMEBUFFER, 0u);
    }

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

        ioEngine->trhandle = glGetTextureSamplerHandleARB(ioEngine->trtex, ioEngine->sampler);
        glMakeTextureHandleResidentARB(ioEngine->trhandle);
        ioEngine->iirhandle = glGetTextureSamplerHandleARB(ioEngine->iirtex, ioEngine->sampler);
        glMakeTextureHandleResidentARB(ioEngine->iirhandle);
        ioEngine->schandle = glGetTextureSamplerHandleARB(ioEngine->sctex, ioEngine->sampler);
        glMakeTextureHandleResidentARB(ioEngine->schandle);

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
            glMakeTextureHandleResidentARB(chunk.handle);
        }

        ioEngine->chunk_texture.reset(0u);
        glCreateTextures(GL_TEXTURE_3D, 1, ioEngine->chunk_texture.get());
        glTextureStorage3D(ioEngine->chunk_texture, 1, GL_R8,
                           1u << engine_t::kLog2ChunkSize,
                           1u << engine_t::kLog2ChunkSize,
                           1u << engine_t::kLog2ChunkSize);

        ioEngine->chunk_handle = glGetTextureSamplerHandleARB(ioEngine->chunk_texture, ioEngine->sampler);
    }

    if (!ioEngine->lock_load_area)
        ioEngine->campos_chunk_index = ComputeChunkIndex(WS_to_VS(ioEngine->campos()));

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

void EngineRunFrame(engine_t* ioEngine, input_t const* iInput)
{
    //static constexpr float kFrameTime = 0.016f;
    const float kFrameTime = iInput->time_step;

    if (false)
    {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(1ms);
    }

#if 0
    std::cout << "input time " << iInput->time_step << std::endl;
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
            numtk::Vec3_t const wscampos = ioEngine->campos();
            int const wscampos_loc = glGetUniformLocation(ioEngine->shader_program, "iWSCamPos");
            if (wscampos_loc >= 0)
                glUniform3fv(wscampos_loc, 1, &wscampos[0]);

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

            numtk::Vec3_t const sundir = {
                0.0, //std::cos(2.9f) * std::sin(1.3f),
                1.0, //std::cos(1.3f),
                0.0, //std::sin(2.9f) * std::sin(1.3f)
            };
            int const sundir_loc = glGetUniformLocation(ioEngine->shader_program, "iSunDir");
            if (sundir_loc >= 0)
                glUniform3fv(sundir_loc, 1, &sundir[0]);

            int const trhandle_loc = glGetUniformLocation(ioEngine->shader_program, "trtex");
            if (trhandle_loc >= 0)
                glUniformHandleui64ARB(trhandle_loc, ioEngine->trhandle);

            int const iirhandle_loc = glGetUniformLocation(ioEngine->shader_program, "iirtex");
            if (iirhandle_loc >= 0)
                glUniformHandleui64ARB(iirhandle_loc, ioEngine->iirhandle);

            int const schandle_loc = glGetUniformLocation(ioEngine->shader_program, "sctex");
            if (schandle_loc >= 0)
                glUniformHandleui64ARB(schandle_loc, ioEngine->schandle);
        }

        glBindBufferBase(GL_UNIFORM_BUFFER, 0u, ioEngine->atmosBuffer);

        {
            // SH preparation
            static const numtk::Vec3_t w[6] {
                {-1.f, 0.f, 0.f},
                {1.f, 0.f, 0.f},
                {0.f, -1.f, 0.f},
                {0.f, 1.f, 0.f},
                {0.f, 0.f, -1.f},
                {0.f, 0.f, 1.f}
            };

            const float tt = iInput->time_total;
            const numtk::Vec3_t Lsun = numtk::vec3_normalise({
                .2f,
                .4f,// * std::cos(tt * 0.5f),
                .4f,// * std::sin(tt * 0.5f)
            });
            const numtk::Vec3_t Csun =
                numtk::vec3_float_mul({ .9f, .8f, .4f },
                                      3.f);
            const numtk::Vec3_t Csky =
                numtk::vec3_float_mul({ .45f, .5f, .7f },
                                      std::max(0.f, 5.f));// * std::cos(tt*0.5f) + 0.25f));

            numtk::Vec3_t Li[6];

            numtk::SH2nd_t sh_normals[6];
            numtk::SH2nd_t sh_reduced[3]{
                numtk::SH2nd_t{},
                numtk::SH2nd_t{},
                numtk::SH2nd_t{}
            };

            static const float weight = 1.f;
            for (int i = 0; i < 6; ++i)
            {
                numtk::Vec3_t fw =
                    numtk::vec3_add(
                        numtk::vec3_float_mul(Csun,
                                              std::max(0.f, numtk::vec3_dot(w[i], Lsun))),
                        numtk::vec3_float_mul(Csky, w[i][1] * 0.5f + 0.5f)
                    );
                Li[i] = fw;

                numtk::SH2nd_t wsh = numtk::sh_second_order(w[i]);
                sh_normals[i] = wsh;

                sh_reduced[0] = numtk::sh2nd_add(sh_reduced[0],
                                                 numtk::sh2nd_float_mul(wsh, fw[0] * weight));
                sh_reduced[1] = numtk::sh2nd_add(sh_reduced[1],
                                                 numtk::sh2nd_float_mul(wsh, fw[1] * weight));
                sh_reduced[2] = numtk::sh2nd_add(sh_reduced[2],
                                                 numtk::sh2nd_float_mul(wsh, fw[2] * weight));
            }

            int loc;
            loc = glGetUniformLocation(ioEngine->shader_program, "iSHBuffer_red");
            glUniform1fv(loc, sizeof(numtk::SH2nd_t), (float*)&sh_reduced[0]);
            loc = glGetUniformLocation(ioEngine->shader_program, "iSHBuffer_green");
            glUniform1fv(loc, sizeof(numtk::SH2nd_t), (float*)&sh_reduced[1]);
            loc = glGetUniformLocation(ioEngine->shader_program, "iSHBuffer_blue");
            glUniform1fv(loc, sizeof(numtk::SH2nd_t), (float*)&sh_reduced[2]);
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
            //glMakeTextureHandleResidentARB(chunk.handle);

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

            //glMakeTextureHandleNonResidentARB(chunk.handle);
        }

        glUseProgram(0u);
    }
}


}

/*

ts : standard time decimal hours
SM : standard meridian (radians)
L : longitude (radians)
J : julian date (integer in [1, 365])

t : solar time decimal hours
t = ts + 0.170*sin(4*pi*(J - 80) / 373) - 0.129*sin(2*pi*(J - 8) / 355) + 12 * (SM - L) / pi

delta : solar declination angle
delta = 0.4093 * sin(2*pi*(J-81)/368)

thetaS : solar angle from zenith
thetaS = pi/2 - arcsin(sin(lat)*sin(delta) - cos(lat)*cos(delta)*cos(pi*t/12))

phiS : solar azimuth
phiS = arctan((-cos(delta) * sin(pi*t/12)) / (cos(lat)*sin(delta) - sin(lat)*cos(delta)*cos(pi*t/12)))


lambda, K, S0, S1, S2, SunRad, ko, kwa, kg
380, 0.650, 63.4, 38.5, 3, 16559.0, /, /, /
390, 0.653, 65.8, 35, 1.2, 16233.7, /, /, /
400, 0.656, 94.8, 43.4, -1.1, 21127.5, /, /, /
410, 0.658, 104.8, 46.3, -0.5, 25888.2, /, /, /
420, 0.661, 105.9, 43.9, -0.7, 25829.1, /, /, /
430, 0.662, 96.8, 37.1, -1.2, 24232.3, /, /, /
440, 0.663, 113.9, 36.7, -2.6, 26760.5, /, /, /
450, 0.666, 125.6, 35.9, -2.9, 29658.3, 0.3, /, /
460, 0.667, 125.5, 32.6, -2.8, 30545.4, 0.6, /, /
470, 0.669, 121.3, 27.9, -2.6, 30057.5, 0.9, /, /
480, 0.670, 121.3, 24.3, -2.6, 30663.7, 1.4, /, /
490, 0.671, 113.5, 20.1, -1.8, 28830.4, 2.1, /, /
500, 0.672, 113.1, 16.2, -1.5, 28712.1, 3.0, /, /
510, 0.673, 110.8, 13.2, -1.3, 27835.0, 4.0, /, /
520, 0.674, 106.5, 8.6, -1.2, 27100.6, 4.8, /, /
530, 0.676, 108.8, 6.1, -1, 27233.6, 6.3, /, /
540, 0.677, 105.3, 4.2, -0.5, 26361.3, 7.5, /, /
550, 0.678, 104.4, 1.9, -0.3, 25503.8, 8.5, /, /
560, 0.679, 100, 0, 0, 25060.2, 10.3, /, /
570, 0.679, 96, -1.6, 0.2, 25311.6, 12, /, /
580, 0.680, 95.1, -3.5, 0.5, 25355.9, 12, /, /
590, 0.681, 89.1, -3.5, 2.1, 25134.2, 11.5, /, /
600, 0.682, 90.5, -5.8, 3.2, 24631.5, 12.5, /, /
610, 0.682, 90.3, -7.2, 4.1, 24173.2, 12, /, /
620, 0.683, 88.4, -8.6, 4.7, 23685.3, 10.5, /, /
630, 0.684, 84, -9.5, 5.1, 23212.1, 9, /, /
640, 0.684, 85.1, -10.9, 6.7, 22827.7, 7.9, /, /
650, 0.685, 81.9, -10.7, 7.3, 22339.8, 6.7, /, /
660, 0.685, 82.6, -12, 8.6, 21970.2, 5.7, /, /
670, 0.685, 84.9, -14, 9.8, 21526.7, 4.8, /, /
680, 0.686, 81.3, -13.6, 10.2, 21097.9, 3.6, /, /
690, 0.686, 71.9, -12, 8.3, 20728.3, 2.8, 1.6, /
700, 0.687, 74.3, -13.3, 9.6, 20240.4, 2.3, 2.4, /
710, 0.687, 76.4, -12.9, 8.5, 19870.8, 1.8, 1.25, /
720, 0.688, 63.3, -10.6, 7, 19427.2, 1.4, 100, /
730, 0.688, 71.7, -11.6, 7.6, 19072.4, 1.1, 87, /
740, 0.689, 77, -12.2, 8, 18628.9, 1, 6.1, /
750, 0.689, 65.2, -10.2, 6.7, 18259.2, 0.9, 0.1, /
760, 0.689, 47.7, -7.8, 5.2, /, 0.7, 1e-03, 3.0,
770, 0.689, 68.6, -11.2, 7.4, /, 0.4, 1e-03, 0.21
780, 0.689, 65, -10.4, 6.8, /, /, 0.06, /

 */


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
    #include "voxeltraversal.frag.glsl.inc"
};

oglbase::ShaderSources_t const layeredgeom{ R"__lstr__(
#version 430 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
uniform float layer;
void main() {
gl_Position = gl_in[0].gl_Position;
gl_Layer = int(layer);
EmitVertex();
gl_Position = gl_in[1].gl_Position;
gl_Layer = int(layer);
EmitVertex();
gl_Position = gl_in[2].gl_Position;
gl_Layer = int(layer);
EmitVertex();
}
)__lstr__" };

oglbase::ShaderSources_t const skytrfrag{
    #include "transmittance.frag.glsl.inc"
};

oglbase::ShaderSources_t const skydirfrag{
    #include "directirradiance.frag.glsl.inc"
};

oglbase::ShaderSources_t const skysscatfrag{
    #include "singlescattering.frag.glsl.inc"
};

oglbase::ShaderSources_t const skyscatdfrag{
    #include "scatteringdensity.frag.glsl.inc"
};

oglbase::ShaderSources_t const skyiirfrag{
    #include "indirectirradiance.frag.glsl.inc"
};

oglbase::ShaderSources_t const skymscatfrag{
    #include "multiplescattering.frag.glsl.inc"
};
