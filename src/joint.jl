abstract JointType{T<:Real}

immutable Joint{T<:Real}
    name::String
    frameBefore::CartesianFrame3D
    frameAfter::CartesianFrame3D
    jointType::JointType{T}

    Joint(name::String, jointType::JointType{T}) = new(name, CartesianFrame3D(string("before_", name)), CartesianFrame3D(string("after_", name)), jointType)
end
Joint{T}(name::String, jointType::JointType{T}) = Joint{T}(name, jointType)
show(io::IO, joint::Joint) = print(io, "Joint \"$(joint.name)\": $(joint.jointType)")
showcompact(io::IO, joint::Joint) = print(io, "$(joint.name)")

immutable QuaternionFloating{T<:Real}<:JointType{T}
end
show(io::IO, jt::QuaternionFloating) = print(io, "Quaternion floating joint")
rand{T}(::Type{QuaternionFloating{T}}) = QuaternionFloating{T}()

@inline function joint_transform{T<:Real}(j::Joint, jt::QuaternionFloating, q::AbstractVector{T})
    length(q) == 7 || error("q has wrong size")
    @inbounds rot = Quaternion(q[1], q[2], q[3], q[4])
    Quaternions.normalize(rot)
    @inbounds trans = SVector{3}(q[5], q[6], q[7])
    return Transform3D{T}(j.frameAfter, j.frameBefore, rot, trans)
end

function motion_subspace{T<:Real}(j::Joint, jt::QuaternionFloating, q::AbstractVector{T})
    angular = hcat(eye(SMatrix{3, 3, T}), zeros(SMatrix{3, 3, T}))
    linear = hcat(zeros(SMatrix{3, 3, T}), eye(SMatrix{3, 3, T}))
    return GeometricJacobian(j.frameAfter, j.frameBefore, j.frameAfter, angular, linear)
end

num_positions(j::Joint, jt::QuaternionFloating) = 7::Int64
num_velocities(j::Joint, jt::QuaternionFloating) = 6::Int64
@inline bias_acceleration{J, T<:Real}(j::Joint{J}, jt::QuaternionFloating{J}, q::AbstractVector{T}, v::AbstractVector{T}) = zero(SpatialAcceleration{promote_type(J, T)}, j.frameAfter, j.frameBefore, j.frameAfter)

function configuration_derivative_to_velocity(j::Joint, jt::QuaternionFloating, vOut::AbstractVector, q::AbstractVector, q̇::AbstractVector)
    length(q) == 7 || error("q has wrong size")
    length(q̇) == 7 || error("q̇ has wrong size")
    @inbounds quat = Quaternion(q[1], q[2], q[3], q[4])
    Quaternions.normalize(quat)
    @inbounds quatdot = Quaternion(q̇[1], q̇[2], q̇[3], q̇[4])
    @inbounds posdot = SVector{3}(q̇[5], q̇[6], q̇[7])
    linear = rotate(posdot, inv(quat))
    angularQuat = 2 * inv(quat) * quatdot
    @inbounds vOut[1] = angularQuat.v1
    @inbounds vOut[2] = angularQuat.v2
    @inbounds vOut[3] = angularQuat.v3
    @inbounds vOut[4] = linear[1]
    @inbounds vOut[5] = linear[2]
    @inbounds vOut[6] = linear[3]
end

function velocity_to_configuration_derivative(j::Joint, jt::QuaternionFloating, q̇Out::AbstractVector, q::AbstractVector, v::AbstractVector)
    length(q) == 7 || error("q has wrong size")
    length(v) == 6 || error("v has wrong size")
    @inbounds quat = Quaternion(q[1], q[2], q[3], q[4])
    Quaternions.normalize(quat)
    @inbounds ωQuat = Quaternion(0, v[1], v[2], v[3])
    @inbounds linear = SVector{3}(v[4], v[5], v[6])
    quatdot = 1/2 * quat * ωQuat
    posdot = rotate(linear, quat)
    @inbounds q̇Out[1] = quatdot.s
    @inbounds q̇Out[2] = quatdot.v1
    @inbounds q̇Out[3] = quatdot.v2
    @inbounds q̇Out[4] = quatdot.v3
    @inbounds q̇Out[5] = posdot[1]
    @inbounds q̇Out[6] = posdot[2]
    @inbounds q̇Out[7] = posdot[3]
    nothing
end

function zero_configuration{T<:Real}(j::Joint, jt::QuaternionFloating, ::Type{T})
    return [one(T); zeros(T, 6)]
end
function rand_configuration{T<:Real}(j::Joint, jt::QuaternionFloating, ::Type{T})
    quat = convert(Quaternion{T}, nquatrand())
    return [quat.s; quat.v1; quat.v2; quat.v3; rand(T, 3)]
end

function joint_twist{T<:Real}(j::Joint, jt::QuaternionFloating, q::AbstractVector{T}, v::AbstractVector{T})
    length(v) == 6 || error("v has wrong size")
    @inbounds ret = Twist(j.frameAfter, j.frameBefore, j.frameAfter, SVector{3}(v[1], v[2], v[3]), SVector{3}(v[4], v[5], v[6]))
    ret
end

abstract OneDegreeOfFreedomFixedAxis{T<:Real}<:JointType{T}

immutable Prismatic{T<:Real}<:OneDegreeOfFreedomFixedAxis{T}
    translation_axis::SVector{3, T}
end
# Prismatic{T}(rotation_axis::SVector{3, T}) = Prismatic{T}(rotation_axis)
show(io::IO, jt::Prismatic) = print(io, "Prismatic joint with axis $(jt.translation_axis)")
function rand{T}(::Type{Prismatic{T}})
    axis = rand(SVector{3, T})
    Prismatic(axis / norm(axis))
end

@inline joint_transform{T1<:Real, T2}(j::Joint, jt::Prismatic{T2}, q::AbstractVector{T1}) = Transform3D(j.frameAfter, j.frameBefore, q[1] * jt.translation_axis)

function joint_twist{T<:Real}(j::Joint, jt::Prismatic, q::AbstractVector{T}, v::AbstractVector{T})
    return Twist(j.frameAfter, j.frameBefore, j.frameAfter, zeros(SVector{3, T}), jt.translation_axis * v[1])
end

function motion_subspace{T<:Real}(j::Joint, jt::Prismatic, q::AbstractVector{T})
    angular = zeros(SMatrix{3, 1, T})
    linear = SMatrix{3, 1, T}(jt.translation_axis)
    return GeometricJacobian(j.frameAfter, j.frameBefore, j.frameAfter, angular, linear)
end

immutable Revolute{T<:Real}<:OneDegreeOfFreedomFixedAxis{T}
    rotation_axis::SVector{3, T}
end
# Revolute{T}(rotation_axis::SVector{3, T}) = Revolute{T}(rotation_axis)
show(io::IO, jt::Revolute) = print(io, "Revolute joint with axis $(jt.rotation_axis)")
function rand{T}(::Type{Revolute{T}})
    axis = rand(SVector{3, T})
    Revolute(axis / norm(axis))
end

@inline function joint_transform{T1, T2}(j::Joint, jt::Revolute{T2}, q::AbstractVector{T1})
    T = promote_type(T1, T2)
    arg = q[1] / T(2)
    s = sin(arg)
    axis = jt.rotation_axis
    rot = Quaternion(cos(arg), s * axis[1], s * axis[2], s * axis[3], true)
    Transform3D(j.frameAfter, j.frameBefore, rot)
end

function joint_twist{T<:Real}(j::Joint, jt::Revolute, q::AbstractVector{T}, v::AbstractVector{T})
    return Twist(j.frameAfter, j.frameBefore, j.frameAfter, jt.rotation_axis * v[1], zeros(SVector{3, T}))
end

function motion_subspace{T<:Real}(j::Joint, jt::Revolute, q::AbstractVector{T})
    angular = SMatrix{3, 1, T}(jt.rotation_axis)
    linear = zeros(SMatrix{3, 1, T})
    return GeometricJacobian(j.frameAfter, j.frameBefore, j.frameAfter, angular, linear)
end

num_positions(j::Joint, jt::OneDegreeOfFreedomFixedAxis) = 1::Int64
num_velocities(j::Joint, jt::OneDegreeOfFreedomFixedAxis) = 1::Int64
zero_configuration{T<:Real}(j::Joint, jt::OneDegreeOfFreedomFixedAxis, ::Type{T}) = [zero(T)]
rand_configuration{T<:Real}(j::Joint, jt::OneDegreeOfFreedomFixedAxis, ::Type{T}) = [rand(T)]
@inline bias_acceleration{J, T<:Real}(j::Joint{J}, jt::OneDegreeOfFreedomFixedAxis{J}, q::AbstractVector{T}, v::AbstractVector{T}) = zero(SpatialAcceleration{promote_type(J, T)}, j.frameAfter, j.frameBefore, j.frameAfter)
function configuration_derivative_to_velocity(j::Joint, jt::OneDegreeOfFreedomFixedAxis, vOut::AbstractVector, q::AbstractVector, q̇::AbstractVector)
    copy!(vOut, q̇)
    nothing
end
function velocity_to_configuration_derivative(j::Joint, jt::OneDegreeOfFreedomFixedAxis, q̇Out::AbstractVector, q::AbstractVector, v::AbstractVector)
    copy!(q̇Out, v)
    nothing
end


immutable Fixed{T<:Real}<:JointType{T}
end
show(io::IO, jt::Fixed) = print(io, "Fixed joint")
rand{T}(::Type{Fixed{T}}) = Fixed{T}()
@inline joint_transform{T}(j::Joint, jt::Fixed, q::AbstractVector{T}) = Transform3D(T, j.frameAfter, j.frameBefore)
function joint_twist{T<:Real}(j::Joint, jt::Fixed, q::AbstractVector{T}, v::AbstractVector{T})
    zero(Twist{T}, j.frameAfter, j.frameBefore, j.frameAfter)
end
function motion_subspace{T<:Real}(j::Joint, jt::Fixed, q::AbstractVector{T})
    GeometricJacobian(j.frameAfter, j.frameBefore, j.frameAfter, zeros(SMatrix{3, 0, T}), zeros(SMatrix{3, 0, T}))
end
num_positions(j::Joint, jt::Fixed) = 0::Int64
num_velocities(j::Joint, jt::Fixed) = 0::Int64
zero_configuration{T<:Real}(j::Joint, jt::Fixed, ::Type{T}) = zeros(T, 0)
rand_configuration{T<:Real}(j::Joint, jt::Fixed, ::Type{T}) = zeros(T, 0)
@inline bias_acceleration{J, T<:Real}(j::Joint{J}, jt::Fixed{J}, q::AbstractVector{T}, v::AbstractVector{T}) = zero(SpatialAcceleration{promote_type(J, T)}, j.frameAfter, j.frameBefore, j.frameAfter)
function configuration_derivative_to_velocity(j::Joint, jt::Fixed, vOut::AbstractVector, q::AbstractVector, q̇::AbstractVector)
    copy!(vOut, q̇)
    nothing
end
function velocity_to_configuration_derivative(j::Joint, jt::Fixed, q̇Out::AbstractVector, q::AbstractVector, v::AbstractVector)
    copy!(q̇Out, v)
    nothing
end

"""
Given a function signature f(args...), generates a function
f(joint::Joint, args...) that delegates to a function
f(joint::Joint, jointType::JointType, args...)
which should be implemented by the implementor of a joint type
"""
macro joint_type_dependent_function(signature)
    functionCall = signature.args[1]
    returnType = signature.args[2]
    functionWithParams = functionCall.args[1]
    functionName = functionWithParams.args[1]
    functionArgs = functionCall.args[2:end]
    argsWithoutType = [arg.args[1] for arg in functionArgs]
    joint = argsWithoutType[1]
    J = functionArgs[1].args[2].args[2]
    argsWithoutType = argsWithoutType[2:end] # skip joint
    eval(quote
        function $(functionWithParams)($(functionArgs...))
            ret::$(returnType)
            jointType = $(joint).jointType
            if isa(jointType, Fixed{J})
            ret = $functionName($joint, $(joint).jointType::Fixed{$(J)}, $(argsWithoutType...))
            elseif isa(jointType, Revolute{J})
                ret = $functionName($joint, $(joint).jointType::Revolute{$(J)}, $(argsWithoutType...))
            elseif isa(jointType, Prismatic{J})
                ret = $functionName($joint, $(joint).jointType::Prismatic{$(J)}, $(argsWithoutType...))
            elseif isa(jointType, QuaternionFloating{J})
                ret = $functionName($joint, $(joint).jointType::QuaternionFloating{$(J)}, $(argsWithoutType...))
            else
                ret = $(returnType)()
            end
            ret
        end
    end)
end

@joint_type_dependent_function joint_transform{J, X}(j::Joint{J}, q::AbstractVector{X})::Transform3D{promote_type(J, X)}
@joint_type_dependent_function motion_subspace{J, X}(j::Joint{J}, q::AbstractVector{X})::GeometricJacobian{promote_type(J, X)}
@joint_type_dependent_function num_positions{J}(j::Joint{J})::Int64
@joint_type_dependent_function num_velocities{J}(j::Joint{J})::Int64
@joint_type_dependent_function bias_acceleration{J, X}(j::Joint{J}, q::AbstractVector{X}, v::AbstractVector{X})::SpatialAcceleration{promote_type(J, X)}
@joint_type_dependent_function configuration_derivative_to_velocity{J, T, X}(j::Joint{J}, vOut::AbstractVector{T}, q::AbstractVector{X}, q̇::AbstractVector{X})::Void
@joint_type_dependent_function velocity_to_configuration_derivative{J, T, X}(j::Joint{J}, q̇Out::AbstractVector{T}, q::AbstractVector{X}, v::AbstractVector{X})::Void
@joint_type_dependent_function zero_configuration{J, X}(j::Joint{J}, t::Type{X})::Vector{promote_type(J, X)}
@joint_type_dependent_function rand_configuration{J, X}(j::Joint{J}, t::Type{X})::Vector{promote_type(J, X)}
@joint_type_dependent_function joint_twist{J, X}(j::Joint{J}, q::AbstractVector{X}, v::AbstractVector{X})::Twist{promote_type(J, X)}


num_positions(itr) = reduce((val, joint) -> val + num_positions(joint), 0, itr)
num_velocities(itr) = reduce((val, joint) -> val + num_velocities(joint), 0, itr)
