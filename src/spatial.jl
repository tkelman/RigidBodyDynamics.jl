immutable SpatialInertia{T<:Real}
    frame::CartesianFrame3D
    moment::SMatrix{3, 3, T, 9}
    crossPart::SVector{3, T} # mass times center of mass
    mass::T
end

convert{T}(::Type{SpatialInertia{T}}, inertia::SpatialInertia{T}) = inertia

function convert{T<:Real}(::Type{SpatialInertia{T}}, inertia::SpatialInertia)
    SpatialInertia(inertia.frame, convert(SMatrix{3, 3, T}, inertia.moment), convert(SVector{3, T}, inertia.crossPart), convert(T, inertia.mass))
end

function convert{T}(::Type{SMatrix{6, 6, T}}, inertia::SpatialInertia)
    J = inertia.moment
    C = vector_to_skew_symmetric(inertia.crossPart)
    m = inertia.mass
    [J  C; C' m * eye(SMatrix{3, 3, T})]
end

convert{T<:Matrix}(::Type{T}, inertia::SpatialInertia) = convert(T, convert(SMatrix{6, 6, eltype(T)}, inertia))

Array{T}(inertia::SpatialInertia{T}) = convert(Matrix{T}, inertia)  # TODO: clean up

center_of_mass(inertia::SpatialInertia) = Point3D(inertia.frame, inertia.crossPart / inertia.mass)

function show(io::IO, inertia::SpatialInertia)
    println(io, "SpatialInertia expressed in \"$(name(inertia.frame))\":")
    println(io, "mass: $(inertia.mass)")
    println(io, "center of mass: $(center_of_mass(inertia))")
    print(io, "moment of inertia:\n$(inertia.moment)")
end

zero{T}(::Type{SpatialInertia{T}}, frame::CartesianFrame3D) = SpatialInertia(frame, zeros(SMatrix{3, 3, T}), zeros(SVector{3, T}), zero(T))

function isapprox(x::SpatialInertia, y::SpatialInertia; atol = 1e-12)
    x.frame == y.frame && isapprox(x.moment, y.moment; atol = atol) && isapprox(x.crossPart, y.crossPart; atol = atol) && isapprox(x.mass, y.mass; atol = atol)
end

function (+){T}(inertia1::SpatialInertia{T}, inertia2::SpatialInertia{T})
    framecheck(inertia1.frame, inertia2.frame)
    moment = inertia1.moment + inertia2.moment
    crossPart = inertia1.crossPart + inertia2.crossPart
    mass = inertia1.mass + inertia2.mass
    SpatialInertia(inertia1.frame, moment, crossPart, mass)
end

function vector_to_skew_symmetric{T}(v::SVector{3, T})
    @SMatrix [zero(T) -v[3] v[2];
              v[3] zero(T) -v[1];
              -v[2] v[1] zero(T)]
end

function cross{N, T}(a::SVector{3, T}, B::SMatrix{3, N, T})
    vector_to_skew_symmetric(a) * B
    # SMatrix(map((col) -> cross(a, SVector(col)).values::Tuple{T, T, T}, B.values))::SMatrix{3, N, T} # way slower
end

function vector_to_skew_symmetric_squared(a::SVector{3})
    aSq1 = a[1] * a[1]
    aSq2 = a[2] * a[2]
    aSq3 = a[3] * a[3]
    b11 = -aSq2 - aSq3
    b12 = a[1] * a[2]
    b13 = a[1] * a[3]
    b22 = -aSq1 - aSq3
    b23 = a[2] * a[3]
    b33 = -aSq1 - aSq2
    @SMatrix [b11 b12 b13;
              b12 b22 b23;
              b13 b23 b33]
end

function transform{I, T}(inertia::SpatialInertia{I}, t::Transform3D{T})
    framecheck(t.from, inertia.frame)
    S = promote_type(I, T)

    if t.from == t.to
        ret = inertia
    elseif inertia.mass == zero(I)
        ret = zero(SpatialInertia{S}, t.to)
    else
        J = convert(SMatrix{3, 3, S}, inertia.moment)
        m = convert(S, inertia.mass)
        c = convert(SVector{3, S}, inertia.crossPart)

        R = RotMatrix(convert(Quat{S}, t.rot)).mat
        p = convert(SVector{3, S}, t.trans)

        cnew = R * c
        Jnew = vector_to_skew_symmetric_squared(cnew)
        cnew += m * p
        Jnew -= vector_to_skew_symmetric_squared(cnew)
        Jnew /= m
        Jnew += R * J * R'
        ret = SpatialInertia(t.to, Jnew, cnew, m)
    end
    ret
end

function rand{T}(::Type{SpatialInertia{T}}, frame::CartesianFrame3D)
    # Try to generate a random but physical moment of inertia
    # by constructing it from its eigendecomposition
    
    Q = RotMatrix(RodriguesVec(rand(), rand(), rand())).mat
    principalMoments = Vector{T}(3)

    # Scale the inertias to make the length scale of the
    # equivalent inertial ellipsoid roughly ~1 unit
    principalMoments[1:2] = rand(2) / 10.

    # Ensure that the principal moments of inertia obey the triangle
    # inequalities:
    # http://www.mathworks.com/help/physmod/sm/mech/vis/about-body-color-and-geometry.html
    lb = abs(principalMoments[1] - principalMoments[2])
    ub = principalMoments[1] + principalMoments[2]
    principalMoments[3] = rand() * (ub - lb) + lb

    # Construct the moment of inertia tensor
    J = SMatrix{3, 3, T}(Q * diagm(principalMoments) * Q')

    # Construct the inertia in CoM frame
    comFrame = CartesianFrame3D("com")
    spatialInertia = SpatialInertia(comFrame, J, zeros(SVector{3, T}), rand(T))

    # Put the center of mass at a random offset
    comFrameToDesiredFrame = Transform3D(comFrame, frame, rand(SVector{3, T}) - 0.5)
    transform(spatialInertia, comFrameToDesiredFrame)
end

immutable Twist{T<:Real}
    # describes motion of body w.r.t. base, expressed in frame
    body::CartesianFrame3D
    base::CartesianFrame3D
    frame::CartesianFrame3D
    angular::SVector{3, T}
    linear::SVector{3, T}
end
Twist{T<:Real}(body::CartesianFrame3D, base::CartesianFrame3D, frame::CartesianFrame3D, vec::AbstractVector{T}) = Twist(body, base, frame, SVector(vec[1], vec[2], vec[3]), SVector(vec[4], vec[5], vec[6]))
convert{T<:Real}(::Type{Twist{T}}, t::Twist{T}) = t
convert{T<:Real}(::Type{Twist{T}}, t::Twist) = Twist(t.body, t.base, t.frame, convert(SVector{3, T}, t.angular), convert(SVector{3, T}, t.linear))
convert{T}(::Type{Vector{T}}, twist::Twist{T}) = [twist.angular...; twist.linear...] # TODO: clean up
Array{T}(twist::Twist{T}) = convert(Vector{T}, twist) # TODO: clean up

function show(io::IO, t::Twist)
    print(io, "Twist of \"$(name(t.body))\" w.r.t \"$(name(t.base))\" in \"$(name(t.frame))\":\nangular: $(t.angular), linear: $(t.linear)")
end

function isapprox(x::Twist, y::Twist; atol = 1e-12)
    x.body == y.body && x.base == y.base && x.frame == y.frame && isapprox(x.angular, y.angular; atol = atol) && isapprox(x.linear, y.linear; atol = atol)
end


function (+)(twist1::Twist, twist2::Twist)
    framecheck(twist1.frame, twist2.frame)
    if twist1.body == twist2.base
        return Twist(twist2.body, twist1.base, twist1.frame, twist1.angular + twist2.angular, twist1.linear + twist2.linear)
    elseif twist1.base == twist2.body
        return Twist(twist1.body, twist2.base, twist1.frame, twist1.angular + twist2.angular, twist1.linear + twist2.linear)
    else
        throw(ArgumentError("frame mismatch"))
    end
end
(-)(t::Twist) = Twist(t.base, t.body, t.frame, -t.angular, -t.linear)

function transform_spatial_motion(angular::SVector{3}, linear::SVector{3}, rot::Quat, p::SVector{3})
    angular = rot * angular
    linear = rot * linear + cross(p, angular)
    return angular, linear
end

function transform(twist::Twist, transform::Transform3D)
    framecheck(twist.frame, transform.from)
    angular, linear = transform_spatial_motion(twist.angular, twist.linear, transform.rot, transform.trans)
    return Twist(twist.body, twist.base, transform.to, angular, linear)
end

change_base_no_relative_motion(t::Twist, base::CartesianFrame3D) = Twist(t.body, base, t.frame, t.angular, t.linear)
change_body_no_relative_motion(t::Twist, body::CartesianFrame3D) = Twist(body, t.base, t.frame, t.angular, t.linear)
zero{T}(::Type{Twist{T}}, body::CartesianFrame3D, base::CartesianFrame3D, frame::CartesianFrame3D) = Twist(body, base, frame, zeros(SVector{3, T}), zeros(SVector{3, T}))
rand{T}(::Type{Twist{T}}, body::CartesianFrame3D, base::CartesianFrame3D, frame::CartesianFrame3D) = Twist(body, base, frame, rand(SVector{3, T}), rand(SVector{3, T}))

immutable GeometricJacobian{A<:AbstractMatrix}
    body::CartesianFrame3D
    base::CartesianFrame3D
    frame::CartesianFrame3D
    angular::A
    linear::A

    function GeometricJacobian(body::CartesianFrame3D, base::CartesianFrame3D, frame::CartesianFrame3D, angular::A, linear::A)
        @assert size(angular, 1) == 3
        @assert size(linear, 1) == 3
        @assert size(angular, 2) == size(linear, 2)
        new(body, base, frame, angular, linear)
    end
end

function GeometricJacobian{A<:AbstractMatrix}(body::CartesianFrame3D, base::CartesianFrame3D, frame::CartesianFrame3D, angular::A, linear::A)
    GeometricJacobian{A}(body, base, frame, angular, linear)
end

# function GeometricJacobian{A<:AbstractMatrix}(body::CartesianFrame3D, base::CartesianFrame3D, frame::CartesianFrame3D, mat::A) # TODO: remove?
#     @assert size(mat, 1) == 6
#     @inbounds angular = mat[1 : 3, :]
#     @inbounds linear = mat[4 : 6, :]
#     GeometricJacobian(body, base, frame, angular, linear)
# end

convert{A}(::Type{GeometricJacobian{A}}, jac::GeometricJacobian{A}) = jac
convert{A}(::Type{GeometricJacobian{A}}, jac::GeometricJacobian) = GeometricJacobian(jac.body, jac.base, jac.frame, convert(A, jac.angular), convert(A, jac.linear))
Array(jac::GeometricJacobian) = [Array(jac.angular); Array(jac.linear)]

eltype{A}(::Type{GeometricJacobian{A}}) = eltype(A)
num_cols(jac::GeometricJacobian) = size(jac.angular, 2)
angular_part(jac::GeometricJacobian) = jac.angular
linear_part(jac::GeometricJacobian) = jac.linear

function Twist(jac::GeometricJacobian, v::AbstractVector)
    Twist(jac.body, jac.base, jac.frame, convert(SVector{3}, jac.angular * v), convert(SVector{3}, jac.linear * v))
end

(-)(jac::GeometricJacobian) = GeometricJacobian(jac.base, jac.body, jac.frame, -jac.angular, -jac.linear)
function show(io::IO, jac::GeometricJacobian)
    print(io, "GeometricJacobian: body: \"$(name(jac.body))\", base: \"$(name(jac.base))\", expressed in \"$(name(jac.frame))\":\n$(Array(jac))")
end

function hcat(jacobians::GeometricJacobian...)
    frame = jacobians[1].frame
    for j = 2 : length(jacobians)
        framecheck(jacobians[j].frame, frame)
        framecheck(jacobians[j].base, jacobians[j - 1].body)
    end
    angular = hcat((jac.angular::SMatrix for jac in jacobians)...)
    linear = hcat((jac.linear::SMatrix for jac in jacobians)...)
    GeometricJacobian(jacobians[end].body, jacobians[1].base, frame, angular, linear)
end

function transform(jac::GeometricJacobian, transform::Transform3D)
    framecheck(jac.frame, transform.from)
    R = RotMatrix(transform.rot).mat
    T = eltype(R)
    angular = R * jac.angular
    linear = R * jac.linear + cross(transform.trans, angular)
    GeometricJacobian(jac.body, jac.base, transform.to, angular, linear)
end

abstract ForceSpaceElement{T<:Real}

immutable Wrench{T<:Real} <: ForceSpaceElement{T}
    frame::CartesianFrame3D
    angular::SVector{3, T}
    linear::SVector{3, T}
end
Wrench{T}(frame::CartesianFrame3D, vec::AbstractVector{T}) = Wrench{T}(frame, SVector(vec[1], vec[2], vec[3]), SVector(vec[4], vec[5], vec[6]))
convert{T<:Real}(::Type{Wrench{T}}, wrench::Wrench{T}) = wrench
convert{T<:Real}(::Type{Wrench{T}}, wrench::Wrench) = Wrench(wrench.frame, convert(SVector{3, T}, wrench.angular), convert(SVector{3, T}, wrench.linear))

show(io::IO, w::Wrench) = print(io, "Wrench expressed in \"$(name(w.frame))\":\nangular: $(w.angular), linear: $(w.linear)")
zero{T}(::Type{Wrench{T}}, frame::CartesianFrame3D) = Wrench(frame, zeros(SVector{3, T}), zeros(SVector{3, T}))
rand{T}(::Type{Wrench{T}}, frame::CartesianFrame3D) = Wrench(frame, rand(SVector{3, T}), rand(SVector{3, T}))

dot(w::Wrench, t::Twist) = begin framecheck(w.frame, t.frame); return dot(w.angular, t.angular) + dot(w.linear, t.linear) end
dot(t::Twist, w::Wrench) = dot(w, t)

immutable Momentum{T<:Real} <: ForceSpaceElement{T}
    frame::CartesianFrame3D
    angular::SVector{3, T}
    linear::SVector{3, T}
end
Momentum{T}(frame::CartesianFrame3D, vec::AbstractVector{T}) = Momentum{T}(frame, SVector(vec[1], vec[2], vec[3]), SVector(vec[4], vec[5], vec[6]))
convert{T<:Real}(::Type{Wrench{T}}, momentum::Momentum{T}) = momentum
convert{T<:Real}(::Type{Wrench{T}}, momentum::Momentum) = Momentum(momentum.frame, convert(SVector{3, T}, momentum.angular), convert(SVector{3, T}, momentum.linear))

show(io::IO, m::Momentum) = print(io, "Momentum expressed in \"$(name(m.frame))\":\nangular: $(m.angular), linear: $(m.linear)")
zero{T}(::Type{Momentum{T}}, frame::CartesianFrame3D) = Momentum(frame, zeros(SVector{3, T}), zeros(SVector{3, T}))
rand{T}(::Type{Momentum{T}}, frame::CartesianFrame3D) = Momentum(frame, rand(SVector{3, T}), rand(SVector{3, T}))


function transform{F<:ForceSpaceElement}(f::F, transform::Transform3D)
    framecheck(f.frame, transform.from)
    linear = transform.rot * f.linear
    angular = transform.rot * f.angular + cross(transform.trans, linear)
    F(transform.to, angular, linear)
end

function (+){F<:ForceSpaceElement}(f1::F, f2::F)
    framecheck(f1.frame, f2.frame)
    F(f1.frame, f1.angular + f2.angular, f1.linear + f2.linear)
end

function (-){F<:ForceSpaceElement}(f1::F, f2::F)
    framecheck(f1.frame, f2.frame)
    F(f1.frame, f1.angular - f2.angular, f1.linear - f2.linear)
end

(-){F<:ForceSpaceElement}(f::F) = F(f.frame, -f.angular, -f.linear)

Array{F<:ForceSpaceElement}(f::F) = [f.angular...; f.linear...]
isapprox{F<:ForceSpaceElement}(x::F, y::F; atol = 1e-12) = x.frame == y.frame && isapprox(x.angular, y.angular, atol = atol) && isapprox(x.linear, y.linear, atol = atol)

function mul_inertia(J, c, m, ω, v)
    angular = J * ω + cross(c, v)
    linear = m * v - cross(c, ω)
    angular, linear
end

function (*)(inertia::SpatialInertia, twist::Twist)
    framecheck(inertia.frame, twist.frame)
    Momentum(inertia.frame, mul_inertia(inertia.moment, inertia.crossPart, inertia.mass, twist.angular, twist.linear)...)
end


immutable MomentumMatrix{A<:AbstractMatrix}
    frame::CartesianFrame3D
    angular::A
    linear::A

    function MomentumMatrix(frame::CartesianFrame3D, angular::A, linear::A)
        @assert size(angular, 1) == 3
        @assert size(linear, 1) == 3
        @assert size(angular, 2) == size(linear, 2)
        new(frame, angular, linear)
    end
end

function MomentumMatrix{A<:AbstractMatrix}(frame::CartesianFrame3D, angular::A, linear::A)
    MomentumMatrix{A}(frame, angular, linear)
end

# function MomentumMatrix{A<:AbstractMatrix}(frame::CartesianFrame3D, mat::A) # TODO: remove?
#     @assert size(mat, 1) == 6
#     @inbounds angular = mat[1 : 3, :]
#     @inbounds linear = mat[4 : 6, :]
#     MomentumMatrix(frame, angular, linear)
# end

convert{A}(::Type{MomentumMatrix{A}}, mat::MomentumMatrix{A}) = mat
convert{A}(::Type{MomentumMatrix{A}}, mat::MomentumMatrix) = MomentumMatrix(mat.frame, convert(A, mat.angular), convert(A, mat.linear))
Array(mat::MomentumMatrix) = [Array(mat.angular); Array(mat.linear)]

eltype{A}(::Type{MomentumMatrix{A}}) = eltype(A)
num_cols(mat::MomentumMatrix) = size(mat.angular, 2)
angular_part(mat::MomentumMatrix) = mat.angular
linear_part(mat::MomentumMatrix) = mat.linear

show(io::IO, m::MomentumMatrix) = print(io, "MomentumMatrix expressed in \"$(name(m.frame))\":\n$(Array(m))")

function (*)(inertia::SpatialInertia, jac::GeometricJacobian)
    framecheck(inertia.frame, jac.frame)
    Jω = jac.angular
    Jv = jac.linear
    J = inertia.moment
    m = inertia.mass
    c = inertia.crossPart
    angular = J * Jω + cross(c, Jv)
    linear = m * Jv - cross(c, Jω)
    MomentumMatrix(inertia.frame, angular, linear)
end

function hcat(mats::MomentumMatrix...)
    frame = mats[1].frame
    for j = 2 : length(mats)
        framecheck(mats[j].frame, frame)
    end
    angular = hcat((m.angular::SMatrix for m in mats)...)
    linear = hcat((m.linear::SMatrix for m in mats)...)
    MomentumMatrix(frame, angular, linear)
end

function Momentum(mat::MomentumMatrix, v::AbstractVector)
    Momentum(mat.frame, convert(SVector{3}, mat.angular * v), convert(SVector{3}, mat.linear * v))
end

function transform(mat::MomentumMatrix, transform::Transform3D)
    framecheck(mat.frame, transform.from)
    R = RotMatrix(transform.rot).mat
    linear = R * linear_part(mat)
    T = eltype(linear)
    angular = R * angular_part(mat) + cross(transform.trans, linear)
    MomentumMatrix(transform.to, angular, linear)
end

immutable SpatialAcceleration{T<:Real}
    body::CartesianFrame3D
    base::CartesianFrame3D
    frame::CartesianFrame3D
    angular::SVector{3, T}
    linear::SVector{3, T}
end

convert{T<:Real}(::Type{SpatialAcceleration{T}}, accel::SpatialAcceleration{T}) = accel
convert{T<:Real}(::Type{SpatialAcceleration{T}}, accel::SpatialAcceleration) = SpatialAcceleration(accel.body, accel.base, accel.frame, convert(SVector{3, T}, accel.angular), convert(SVector{3, T}, accel.linear))

function SpatialAcceleration(body::CartesianFrame3D, base::CartesianFrame3D, frame::CartesianFrame3D, vec::AbstractVector)
    SpatialAcceleration(body, base, frame, SVector(vec[1], vec[2], vec[3]), SVector(vec[4], vec[5], vec[6]))
end

function SpatialAcceleration(jac::GeometricJacobian, v̇::AbstractVector)
    v̇Fixed = SVector{num_cols(jac)}(v̇)
    SpatialAcceleration(jac.body, jac.base, jac.frame, jac.angular * v̇Fixed, jac.linear * v̇Fixed)
end

function (+)(accel1::SpatialAcceleration, accel2::SpatialAcceleration)
    framecheck(accel1.frame, accel2.frame)
    if accel1.body == accel2.base
        return SpatialAcceleration(accel2.body, accel1.base, accel1.frame, accel1.angular + accel2.angular, accel1.linear + accel2.linear)
    elseif accel1.body == accel2.body && accel1.base == accel2.base
        return SpatialAcceleration(accel1.body, accel1.base, accel1.frame, accel1.angular + accel2.angular, accel1.linear + accel2.linear)
    end
    @assert false
end

(-)(accel::SpatialAcceleration) = SpatialAcceleration(accel.base, accel.body, accel.frame, -accel.angular, -accel.linear)

Array(accel::SpatialAcceleration) = [accel.angular...; accel.linear...]

function show(io::IO, a::SpatialAcceleration)
    print(io, "SpatialAcceleration of \"$(name(a.body))\" w.r.t \"$(name(a.base))\" in \"$(name(a.frame))\":\nangular: $(a.angular), linear: $(a.linear)")
end

function transform(accel::SpatialAcceleration, oldToNew::Transform3D, twistOfCurrentWrtNew::Twist, twistOfBodyWrtBase::Twist)
    # trivial case
    accel.frame == oldToNew.to && return accel

    # frame checks
    framecheck(oldToNew.from, accel.frame)
    framecheck(twistOfCurrentWrtNew.frame, accel.frame)
    framecheck(twistOfCurrentWrtNew.body, accel.frame)
    framecheck(twistOfCurrentWrtNew.base, oldToNew.to)
    framecheck(twistOfBodyWrtBase.frame, accel.frame)
    framecheck(twistOfBodyWrtBase.body, accel.body)
    framecheck(twistOfBodyWrtBase.base, accel.base)

    # spatial motion cross product:
    angular = cross(twistOfCurrentWrtNew.angular, twistOfBodyWrtBase.angular)
    linear = cross(twistOfCurrentWrtNew.linear, twistOfBodyWrtBase.angular)
    linear += cross(twistOfCurrentWrtNew.angular, twistOfBodyWrtBase.linear)

    # add current acceleration:
    angular += accel.angular
    linear += accel.linear

    # transform to new frame
    angular, linear = transform_spatial_motion(angular, linear, oldToNew.rot, oldToNew.trans)

    SpatialAcceleration(accel.body, accel.base, oldToNew.to, angular, linear)
end

zero{T}(::Type{SpatialAcceleration{T}}, body::CartesianFrame3D, base::CartesianFrame3D, frame::CartesianFrame3D) = SpatialAcceleration(body, base, frame, zeros(SVector{3, T}), zeros(SVector{3, T}))
rand{T}(::Type{SpatialAcceleration{T}}, body::CartesianFrame3D, base::CartesianFrame3D, frame::CartesianFrame3D) = SpatialAcceleration(body, base, frame, rand(SVector{3, T}), rand(SVector{3, T}))

function newton_euler(I::SpatialInertia, Ṫ::SpatialAcceleration, T::Twist)
    body = Ṫ.body
    base = Ṫ.base # TODO: should assert that this is an inertial frame somehow
    frame = Ṫ.frame

    framecheck(I.frame, frame)
    framecheck(T.body, body)
    framecheck(T.base, base)
    framecheck(T.frame, frame)

    angular, linear = mul_inertia(I.moment, I.crossPart, I.mass, Ṫ.angular, Ṫ.linear)
    angularMomentum, linearMomentum = mul_inertia(I.moment, I.crossPart, I.mass, T.angular, T.linear)
    angular += cross(T.angular, angularMomentum) + cross(T.linear, linearMomentum)
    linear += cross(T.angular, linearMomentum)
    Wrench(frame, angular, linear)
end

function joint_torque(jac::GeometricJacobian, wrench::Wrench)
    framecheck(jac.frame, wrench.frame)
    jac.angular' * wrench.angular + jac.linear' * wrench.linear
end

function kinetic_energy(I::SpatialInertia, twist::Twist)
    framecheck(I.frame, twist.frame)
    # TODO: should assert that twist.base is an inertial frame somehow
    ω = twist.angular
    v = twist.linear
    J = I.moment
    c = I.crossPart
    m = I.mass
    1/2 * (dot(ω, J * ω) + dot(v, m * v + 2 * cross(ω, c)))
end
