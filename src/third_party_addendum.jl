rotate(x::SMatrix{3}, q::Quaternion) = rotationmatrix_normalized_fsa(q) * x

function rotate(x::SVector{3}, q::Quaternion)
    qret = q * Quaternion(zero(eltype(x)), x[1], x[2], x[3]) * inv(q)
    SVector(qret.v1, qret.v2, qret.v3)
end

angle_axis_proper(q::Quaternion) = angle_proper(q), axis_proper(q)

angle_proper(q::Quaternion) = 2 * acos(real(Quaternions.normalize(q)))

function axis_proper(q::Quaternion)
    q = Quaternions.normalize(q)
    s = sin(angle(q) / 2)
        abs(s) > 0 ?
        [q.v1, q.v2, q.v3] / s :
        [1.0, 0.0, 0.0]
end

function rotationmatrix_normalized_fsa{T}(q::Quaternion{T})
    sx, sy, sz = 2q.s*q.v1, 2q.s*q.v2, 2q.s*q.v3
    xx, xy, xz = 2q.v1^2, 2q.v1*q.v2, 2q.v1*q.v3
    yy, yz, zz = 2q.v2^2, 2q.v2*q.v3, 2q.v3^2
    @SMatrix [one(T)-(yy+zz) xy-sz xz+sy;
              xy+sz one(T)-(xx+zz) yz-sx;
              xz-sy yz+sx one(T)-(xx+yy)]
end

function rpy_to_quaternion(rpy::Vector)
    length(rpy) != 3 && error("wrong size")
    rpy2 = rpy / 2
    s = sin.(rpy2)
    c = cos.(rpy2)
    @inbounds qs = c[1]*c[2]*c[3] + s[1]*s[2]*s[3]
    @inbounds qx = s[1]*c[2]*c[3] - c[1]*s[2]*s[3]
    @inbounds qy = c[1]*s[2]*c[3] + s[1]*c[2]*s[3]
    @inbounds qz = c[1]*c[2]*s[3] - s[1]*s[2]*c[3]
    Quaternion(qs, qx, qy, qz)
end

# TODO: notify StaticArrays maintainer
# TODO: too specific
function (*){N, M, T, L, S}(m::SMatrix{N, M, T, L}, x::Union{Vector{S}, VectorSegment{S}})
    @boundscheck @assert M == size(x, 1)
    ret = zeros(SVector{N, promote_type(T, S)})
    for i = 1 : M
        @inbounds xi = x[i]
        ret = ret + m[:, i] * xi
    end
    ret
end

# # TODO: notify StaticArrays maintainer
# @inline (::Type{SVector{0}}){T}(::VectorSegment{T}) = zeros(SVector{0, T})
