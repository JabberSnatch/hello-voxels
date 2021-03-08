/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * Samuel Bourasseau wrote this file. You can do whatever you want with this
 * stuff. If we meet some day, and you think this stuff is worth it, you can
 * buy me a beer in return.
 * ----------------------------------------------------------------------------
 */

#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace numtk {

using Vec2i_t = std::array<int, 2>;
using Vec3i64_t = std::array<std::int64_t, 3>;
using Vec2_t = std::array<float, 2>;
using Vec3_t = std::array<float, 3>;
using Vec4_t = std::array<float, 4>;
using Mat4_t = std::array<float, 16>;
using Quat_t = std::array<float, 4>;
using dQuat_t = std::array<float, 8>;

constexpr float kPi = 3.1415926535f;

inline Vec2i_t
vec2i_sub(Vec2i_t const& _lhs, Vec2i_t const&_rhs)
{
    return Vec2i_t{ _lhs[0] - _rhs[0], _lhs[1] - _rhs[1] };
}

inline Vec2i_t
vec2i_int_div(Vec2i_t const& _lhs, int _rhs)
{
    return Vec2i_t{ _lhs[0] / _rhs, _lhs[1] / _rhs };
}

inline bool
vec2i_equal(Vec2i_t const& _lhs, Vec2i_t const& _rhs)
{
    return (_lhs[0] == _rhs[0]) && (_lhs[1] == _rhs[1]);
}

inline Vec2_t
vec2_itof(Vec2i_t const& _in)
{
    return Vec2_t{ (float)_in[0], (float)_in[1] };
}

inline Vec3i64_t
vec3i64_add(Vec3i64_t const& _lhs, Vec3i64_t const& _rhs)
{
    return Vec3i64_t{ _lhs[0]+_rhs[0], _lhs[1]+_rhs[1], _lhs[2]+_rhs[2] };
}

inline Vec3i64_t
vec3i64_sub(Vec3i64_t const& _lhs, Vec3i64_t const& _rhs)
{
    return Vec3i64_t{ _lhs[0]-_rhs[0], _lhs[1]-_rhs[1], _lhs[2]-_rhs[2] };
}

inline Vec3i64_t
vec3i64_int_mul(Vec3i64_t const& _lhs, std::int64_t _rhs)
{
    return Vec3i64_t{ _lhs[0]*_rhs, _lhs[1]*_rhs, _lhs[2]*_rhs };
}

inline Vec3i64_t
vec3i64_int_div(Vec3i64_t const& _lhs, std::int64_t _rhs)
{
    return Vec3i64_t{ _lhs[0]/_rhs, _lhs[1]/_rhs, _lhs[2]/_rhs };
}

inline Vec3_t
vec3_from_vec4(Vec4_t const& _o)
{
    return Vec3_t{ _o[0], _o[1], _o[2] };
}

inline Vec3_t
vec3_constant(float _op)
{
    return Vec3_t{ _op, _op, _op };
}

inline Vec3_t
vec3_sub(Vec3_t const& _lhs, Vec3_t const& _rhs)
{
    return Vec3_t{ _lhs[0]-_rhs[0], _lhs[1]-_rhs[1], _lhs[2]-_rhs[2] };
}

inline Vec3_t
vec3_add(Vec3_t const& _lhs, Vec3_t const& _rhs)
{
    return Vec3_t{ _lhs[0]+_rhs[0], _lhs[1]+_rhs[1], _lhs[2]+_rhs[2] };
}

inline Vec3_t
vec3_float_mul(Vec3_t const& _lhs, float _rhs)
{
    return Vec3_t{ _lhs[0]*_rhs, _lhs[1]*_rhs, _lhs[2]*_rhs };
}

inline Vec3_t
vec3_cwise_mul(Vec3_t const& _lhs, Vec3_t const& _rhs)
{
    return Vec3_t{ _lhs[0]*_rhs[0], _lhs[1]*_rhs[1], _lhs[2]*_rhs[2] };
}

inline float
vec3_dot(Vec3_t const& _lhs, Vec3_t const& _rhs)
{
    return _lhs[0]*_rhs[0] + _lhs[1]*_rhs[1] + _lhs[2]*_rhs[2];
}

inline Vec3_t
vec3_cross(Vec3_t const& _lhs, Vec3_t const& _rhs)
{
    return Vec3_t{
        _lhs[1]*_rhs[2] - _lhs[2]*_rhs[1],
        _lhs[2]*_rhs[0] - _lhs[0]*_rhs[2],
        _lhs[0]*_rhs[1] - _lhs[1]*_rhs[0]
    };
}

inline float
vec3_norm(Vec3_t const& _op)
{
    return std::sqrt(vec3_dot(_op, _op));
}

inline Vec3_t
vec3_normalise(Vec3_t const& _op)
{
    return vec3_float_mul(_op, 1.f/std::sqrt(vec3_dot(_op, _op)));
}

inline Vec4_t
vec3_float_concat(Vec3_t const& _lhs, float const& _rhs)
{
    return Vec4_t{ _lhs[0], _lhs[1], _lhs[2], _rhs };
}

inline Vec4_t
vec4_add(Vec4_t const& _lhs, Vec4_t const& _rhs)
{
    return Vec4_t{ _lhs[0] + _rhs[0], _lhs[1] + _rhs[1], _lhs[2] + _rhs[2], _lhs[3] + _rhs[3] };
}

inline Vec4_t
vec4_float_mul(Vec4_t const& _lhs, float _rhs)
{
    return Vec4_t{ _lhs[0] * _rhs, _lhs[1] * _rhs, _lhs[2] * _rhs, _lhs[3] * _rhs};
}

inline float
vec4_dot(Vec4_t const& _lhs, Vec4_t const& _rhs)
{
    return _lhs[0]*_rhs[0] + _lhs[1]*_rhs[1] + _lhs[2]*_rhs[2] + _lhs[3]*_rhs[3];
}

inline Mat4_t
mat4_id()
{
    return Mat4_t{
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    };
}

inline Mat4_t
mat4_col(Vec4_t const& _c0, Vec4_t const& _c1, Vec4_t const& _c2, Vec4_t const& _c3)
{
    return Mat4_t{
        _c0[0], _c0[1], _c0[2], _c0[3],
        _c1[0], _c1[1], _c1[2], _c1[3],
        _c2[0], _c2[1], _c2[2], _c2[3],
        _c3[0], _c3[1], _c3[2], _c3[3]
        };
}

inline Mat4_t
mat4_transpose(Mat4_t const& _op)
{
    return Mat4_t{
        _op[0], _op[4], _op[8], _op[12],
        _op[1], _op[5], _op[9], _op[13],
        _op[2], _op[6], _op[10], _op[14],
        _op[3], _op[7], _op[11], _op[15]
    };
}

inline Vec4_t
mat4_vec4_mul(Mat4_t const& _lhs, Vec4_t const& _rhs)
{
    return Vec4_t{
        _lhs[0]*_rhs[0] + _lhs[4]*_rhs[1] + _lhs[8]*_rhs[2] + _lhs[12]*_rhs[3],
        _lhs[1]*_rhs[0] + _lhs[5]*_rhs[1] + _lhs[9]*_rhs[2] + _lhs[13]*_rhs[3],
        _lhs[2]*_rhs[0] + _lhs[6]*_rhs[1] + _lhs[10]*_rhs[2] + _lhs[14]*_rhs[3],
        _lhs[3]*_rhs[0] + _lhs[7]*_rhs[1] + _lhs[11]*_rhs[2] + _lhs[15]*_rhs[3]
    };
}

inline Mat4_t
mat4_mul(Mat4_t const& _lhs, Mat4_t const& _rhs)
{
    Mat4_t const lt = mat4_transpose(_lhs);
    Vec4_t const& _lc0 = *(Vec4_t*)&lt;
    Vec4_t const& _lc1 = *(((Vec4_t*)&lt) + 1);
    Vec4_t const& _lc2 = *(((Vec4_t*)&lt) + 2);
    Vec4_t const& _lc3 = *(((Vec4_t*)&lt) + 3);

    Vec4_t const& _rc0 = *(Vec4_t*)&_rhs;
    Vec4_t const& _rc1 = *(((Vec4_t*)&_rhs) + 1);
    Vec4_t const& _rc2 = *(((Vec4_t*)&_rhs) + 2);
    Vec4_t const& _rc3 = *(((Vec4_t*)&_rhs) + 3);

    return mat4_col(
        Vec4_t{vec4_dot(_lc0, _rc0), vec4_dot(_lc1, _rc0), vec4_dot(_lc2, _rc0), vec4_dot(_lc3, _rc0)},
        Vec4_t{vec4_dot(_lc0, _rc1), vec4_dot(_lc1, _rc1), vec4_dot(_lc2, _rc1), vec4_dot(_lc3, _rc1)},
        Vec4_t{vec4_dot(_lc0, _rc2), vec4_dot(_lc1, _rc2), vec4_dot(_lc2, _rc2), vec4_dot(_lc3, _rc2)},
        Vec4_t{vec4_dot(_lc0, _rc3), vec4_dot(_lc1, _rc3), vec4_dot(_lc2, _rc3), vec4_dot(_lc3, _rc3)}
    );
}

inline Mat4_t
mat4_rot(Vec3_t const& _eulerDeg)
{
    Vec3_t er = vec3_float_mul(_eulerDeg, 3.1415926536f/180.f);

    Mat4_t x{
        1.f, 0.f, 0.f, 0.f,
        0.f, std::cos(er[0]), std::sin(er[0]), 0.f,
        0.f, -std::sin(er[0]), std::cos(er[0]), 0.f,
        0.f, 0.f, 0.f, 1.f
    };
    Mat4_t y{
        std::cos(er[1]), 0.f, -std::sin(er[1]), 0.f,
        0.f, 1.f, 0.f, 0.f,
        std::sin(er[1]), 0.f, std::cos(er[1]), 0.f,
        0.f, 0.f, 0.f, 1.f
    };
    Mat4_t z{
        std::cos(er[2]), std::sin(er[2]), 0.f, 0.f,
        -std::sin(er[2]), std::cos(er[2]), 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    };

    return mat4_mul(y, mat4_mul(x, z));
}

inline Mat4_t
mat4_from_quat(Quat_t const& _op)
{
     return Mat4_t{
        1.f - 2.f*(_op[2]*_op[2] + _op[3]*_op[3]),
        2.f*(_op[1]*_op[2] + _op[3]*_op[0]),
        2.f*(_op[1]*_op[3] - _op[2]*_op[0]),
        0.f,

        2.f*(_op[1]*_op[2] - _op[3]*_op[0]),
        1.f - 2.f*(_op[1]*_op[1] + _op[3]*_op[3]),
        2.f*(_op[2]*_op[3] + _op[1]*_op[0]),
        0.f,

        2.f*(_op[1]*_op[3] + _op[2]*_op[0]),
        2.f*(_op[2]*_op[3] - _op[1]*_op[0]),
        1.f - 2.f*(_op[1]*_op[1] + _op[2]*_op[2]),
        0.f,

        0.f, 0.f, 0.f, 1.f
    };
}

/*

mat4 perspective_hfov(float n, float f, float alpha, float aspect)
{
	float inverse_tan_half_alpha = 1.0 / tan(alpha * 0.5);
	return mat4(
		inverse_tan_half_alpha, 0.0, 0.0, 0.0,
		0.0, (1.0/aspect)*inverse_tan_half_alpha, 0.0, 0.0,
		0.0, 0.0, (-(f+n))/(f-n), -1.0,
		0.0, 0.0, (-2.0*f*n)/(f-n), 0.0
	);
}

*/

inline Mat4_t
mat4_perspective(float n, float f, float alpha, float how)
{
    const float inv_tan_half_alpha = 1.f / std::tan(alpha * .5f);
    const float inv_fmn = 1.f / (f-n);
    return Mat4_t{
        inv_tan_half_alpha, 0.f, 0.f, 0.f,
        0.f, (1.f/how)*inv_tan_half_alpha, 0.f, 0.f,
        0.f, 0.f, (-(f+n))*inv_fmn, -1.f,
        0.f, 0.f, -2.f*f*n*inv_fmn, 0.f
    };
}

/*

mat4 perspective_inverse_hfov(float n, float f, float alpha, float aspect)
{
	float tan_half_alpha = tan(alpha * 0.5) / 1.0;
	return mat4(
		tan_half_alpha, 0.0, 0.0, 0.0,
		0.0, aspect * tan_half_alpha, 0.0, 0.0,
		0.0, 0.0, 0.0, (f-n)/(-2.0*f*n),
		0.0, 0.0, -1.0, (f+n)/(2.0*f*n)
	);
}

 */

inline Mat4_t
mat4_perspectiveinv(float n, float f, float alpha, float how)
{
	float tan_half_alpha = tan(alpha * 0.5f);
	return Mat4_t{
		tan_half_alpha, 0.0f, 0.0f, 0.0f,
		0.0f, how * tan_half_alpha, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, (f-n)/(-2.0f*f*n),
		0.0f, 0.0f, -1.0f, (f+n)/(2.0f*f*n)
	};
}

inline Mat4_t
mat4_view(Vec3_t const& _f, Vec3_t const& _u, Vec3_t const& _p)
{
    numtk::Vec3_t const l = numtk::vec3_normalise(numtk::vec3_cross(_u, _f));
    numtk::Vec3_t const p = numtk::vec3_float_mul(_p, -1.f);
    numtk::Vec3_t const t{ numtk::vec3_dot(p, l), numtk::vec3_dot(p, _u), numtk::vec3_dot(p, _f) };

    return numtk::mat4_col(numtk::Vec4_t{l[0], _u[0], _f[0], 0.f},
                           numtk::Vec4_t{l[1], _u[1], _f[1], 0.f},
                           numtk::Vec4_t{l[2], _u[2], _f[2], 0.f},
                           numtk::vec3_float_concat(t, 1.f));
}

inline Mat4_t
mat4_viewinv(Vec3_t const& _f, Vec3_t const& _u, Vec3_t const& _p)
{
    numtk::Vec3_t const u = numtk::vec3_float_mul(_u, 1.f);
    numtk::Vec3_t const f = numtk::vec3_float_mul(_f, 1.f);
    numtk::Vec3_t const l = numtk::vec3_normalise(numtk::vec3_cross(u, f));

    return numtk::mat4_col(numtk::vec3_float_concat(l, 0.f),
                           numtk::vec3_float_concat(u, 0.f),
                           numtk::vec3_float_concat(f, 0.f),
                           numtk::vec3_float_concat(_p, 1.f));
}

inline Quat_t
quat_angle_axis(float _angle, numtk::Vec3_t const &_axis)
{
    float const half_angle = _angle / 2.f;
    float const sin_ha = std::sin(half_angle);
    return Quat_t{ std::cos(half_angle), _axis[0] * sin_ha, _axis[1] * sin_ha, _axis[2] * sin_ha };
}

inline Quat_t
quat_add(Quat_t const& _lhs, Quat_t const& _rhs)
{
    return (Quat_t)vec4_add((Vec4_t)_lhs, (Vec4_t)_rhs);
}

inline Quat_t
quat_mul(Quat_t const& _lhs, Quat_t const& _rhs)
{
    // 1 i j k
    // i −1 k −j
    // j −k −1 i
    // k j −i −1

#if 0
    Vec3_t const& v0 = *(Vec3_t const*)&(_lhs[1]);
    Vec3_t const& v1 = *(Vec3_t const*)&(_rhs[1]);

    Vec3_t v = vec3_add(vec3_float_mul(v1, _lhs[0]),
                        vec3_add(vec3_float_mul(v0, _rhs[0]),
                                 vec3_cross(v0, v1)));
    return Quat_t{
        _lhs[0]*_rhs[0] - vec3_dot(*(Vec3_t const*)&(_lhs[1]), *(Vec3_t const*)&(_rhs[1])),
        v[0], v[1], v[2]
    };
#else
    return Quat_t{
        _lhs[0]*_rhs[0] - _lhs[1]*_rhs[1] - _lhs[2]*_rhs[2] - _lhs[3]*_rhs[3],
        _lhs[0]*_rhs[1] + _lhs[1]*_rhs[0] + _lhs[2]*_rhs[3] - _lhs[3]*_rhs[2],
        _lhs[0]*_rhs[2] - _lhs[1]*_rhs[3] + _lhs[2]*_rhs[0] + _lhs[3]*_rhs[1],
        _lhs[0]*_rhs[3] + _lhs[1]*_rhs[2] - _lhs[2]*_rhs[1] + _lhs[3]*_rhs[0]
    };
#endif
}

inline Quat_t
quat_float_mul(Quat_t const& _lhs, float _rhs)
{
    return (Quat_t)vec4_float_mul((Vec4_t)_lhs, _rhs);
}

inline float
quat_norm(Quat_t const& _op)
{
    return std::sqrt(vec4_dot((Vec4_t)_op, (Vec4_t)_op));
}

inline Quat_t
quat_normalise(Quat_t const& _op)
{
    return quat_float_mul(_op, 1.f / quat_norm(_op));
}

inline Quat_t
quat_conjugate(Quat_t const& _op)
{
    return Quat_t{ _op[0], -_op[1], -_op[2], -_op[3] };
}

inline Quat_t
quat_from_euler(numtk::Vec3_t const& _e)
{
#if 1
    Quat_t x = quat_angle_axis(_e[0], {1.f, 0.f, 0.f});
    Quat_t y = quat_angle_axis(_e[1], {0.f, 1.f, 0.f});
    Quat_t z = quat_angle_axis(_e[2], {0.f, 0.f, 1.f});
    return quat_mul(y, quat_mul(x, z));

#else
    numtk::Vec3_t h = numtk::vec3_float_mul(_e, .5f);
    numtk::Vec3_t s{ std::sin(h[0]), std::sin(h[1]), std::sin(h[2]) };
    numtk::Vec3_t c{ std::cos(h[0]), std::cos(h[1]), std::cos(h[2]) };

    return Quat_t{
        c[2]*c[0]*c[1] + s[2]*s[0]*s[1],
        s[2]*c[0]*c[1] - c[2]*s[0]*s[1],
        c[2]*s[0]*c[1] + s[2]*c[0]*s[1],
        c[2]*c[0]*s[1] - s[2]*s[0]*c[1]
    };
#endif
}

inline Quat_t
quat_slerp(Quat_t const& _lhs, Quat_t const& _rhs, float _t)
{
#if 0
    float costh = vec3_dot(*(Vec3_t*)&_lhs[1], *(Vec3_t*)&_rhs[1]);
    float sign = (costh < 0.f) ? -1.f : 1.f;
    float theta = std::acos(costh);
    if (std::abs(theta) < 0.0001f)
        return _rhs;

    float t_theta = _t * theta;
    float sinth = std::sin(theta);
    float f0 = std::sin(theta - t_theta) / sinth;
    float f1 = sign * (std::sin(t_theta) / sinth);
    Quat_t result = quat_add(quat_float_mul(_lhs, f0), quat_float_mul(_rhs, f1));
    return result;
#else

    // A Fast and Accurate Algorithm for Computing SLERP, Eberly 2011
    // Port of https://github.com/CesiumCG/cesium implementation

    static constexpr float kopmu = 1.90110745351730037f;

    /*
    for (var i = 0; i < 7; ++i) {
        var s = i + 1.0;
        var t = 2.0 * s + 1.0;
        u[i] = 1.0 / (s * t);
        v[i] = s / t;
    }

    u[7] = opmu / (8.0 * 17.0);
    v[7] = opmu * 8.0 / 17.0;
    */

    static const float u[8] = {
        1.f / 3.f,
        1.f / 10.f,
        1.f / 21.f,
        1.f / 36.f,
        1.f / 55.f,
        1.f / 78.f,
        1.f / 105.f,
        kopmu / 136.f
    };

    static const float v[8] = {
        1.f / 3.f,
        2.f / 5.f,
        3.f / 7.f,
        4.f / 9.f,
        5.f / 11.f,
        6.f / 13.f,
        7.f / 15.f,
        kopmu * 8.f / 17.f
    };

    float dot = vec4_dot((Vec4_t const&)_lhs, (Vec4_t const&)_rhs);
    float const sign = (dot < 0.f) ? -1.f : 1.f;
    dot = sign * dot;

    float const dotminus1 = dot - 1.f;
    float const d = 1.f - _t;
    float const sqrT = _t * _t;
    float const sqrD = d * d;

    float bT[8];
    float bD[8];
    for (int i = 0; i < 8; ++i)
    {
        bT[i] = (u[i] * sqrT - v[i]) * dotminus1;
        bD[i] = (u[i] * sqrD - v[i]) * dotminus1;
    }

    float const cT = sign * _t * (
        1.f + bT[0] * (
            1.f + bT[1] * (
                1.f + bT[2] * (
                    1.f + bT[3] * (
                        1.f + bT[4] * (
                            1.f + bT[5] * (
                                1.f + bT[6] * (
                                    1.f + bT[7]))))))));

    float const cD = d *(
        1.f + bD[0] * (
            1.f + bD[1] * (
                1.f + bD[2] * (
                    1.f + bD[3] * (
                        1.f + bD[4] * (
                            1.f + bD[5] * (
                                1.f + bD[6] * (
                                    1.f + bD[7]))))))));

    Quat_t result = quat_add(quat_float_mul(_lhs, cD), quat_float_mul(_rhs, cT));
    return result;
#endif
}

inline dQuat_t
dquat(float _angle, Vec3_t _axis, Vec3_t _t)
{
    Quat_t const r = quat_angle_axis(_angle, _axis);
    Quat_t const t{ 0, _t[0], _t[1], _t[2] };
    Quat_t const dual = quat_float_mul(quat_mul(t, r), .5f);
    return dQuat_t{ r[0], r[1], r[2], r[3], dual[0], dual[1], dual[2], dual[3] };
}

inline Quat_t const&
dquat_rotation(dQuat_t const& _op)
{
    return *(Quat_t const*)&_op;
}

} // namespace numtk


