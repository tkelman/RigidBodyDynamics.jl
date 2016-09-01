type JointAndState{M, X}
    joint::Joint{M}
    q::VectorSegment{X}
    v::VectorSegment{X}
end

immutable MechanismState{X<:Real, M<:Real, C<:Real}
    mechanism::Mechanism{M}
    q::Vector{X}
    v::Vector{X}
    frameAfterJointToJointAndState::Dict{CartesianFrame3D, JointAndState{M, X}}
    successorBodyToJointAndState::Dict{RigidBody{M}, JointAndState{M, X}}
    jointTransforms::Dict{Joint{M}, CacheElement{Transform3D{C}}}
    fixedTransformsToParent::Dict{CartesianFrame3D, Transform3D{C}}
    transformsToRoot::Dict{CartesianFrame3D, CacheElement{Transform3D{C}}}
    twists::Dict{RigidBody{M}, CacheElement{Twist{C}}}
    biasAccelerations::Dict{RigidBody{M}, CacheElement{SpatialAcceleration{C}}}
    motionSubspaces::Dict{Joint{M}, CacheElement{GeometricJacobian}} # TODO: fix type
    spatialInertias::Dict{RigidBody{M}, CacheElement{SpatialInertia{C}}}
    crbInertias::Dict{RigidBody{M}, CacheElement{SpatialInertia{C}}}

    function MechanismState(m::Mechanism{M})
        q = Vector{X}(num_positions(m))
        v = Vector{X}(num_velocities(m))
        state = new(m, q, v, Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict())
        state.fixedTransformsToParent[root_frame(m)] = Transform3D{C}(root_frame(m), root_frame(m))
        zero!(state)
        state
    end
end

eltype{X, M, C}(::MechanismState{X, M, C}) = C
show{X, M, C}(io::IO, ::MechanismState{X, M, C}) = print(io, "MechanismState{$X, $M, $C}(…)")
num_positions(state::MechanismState) = length(state.q)
num_velocities(state::MechanismState) = length(state.v)
state_vector_eltype{X, M, C}(state::MechanismState{X, M, C}) = X
mechanism_eltype{X, M, C}(state::MechanismState{X, M, C}) = M
configuration(state::MechanismState, joint::Joint) = view(state.q, state.mechanism.qRanges[joint])
velocity(state::MechanismState, joint::Joint) = view(state.v, state.mechanism.vRanges[joint])
configuration_vector(state::MechanismState) = state.q
velocity_vector(state::MechanismState) = state.v
state_vector(state::MechanismState) = [configuration_vector(state); velocity_vector(state)] # TODO: consider having x be a member and using view for q and v
configuration_vector{T}(state::MechanismState, path::Path{RigidBody{T}, Joint{T}}) = vcat([state.q[state.mechanism.qRanges[joint]] for joint in path.edgeData]...)
velocity_vector{T}(state::MechanismState, path::Path{RigidBody{T}, Joint{T}}) = vcat([state.v[state.mechanism.vRanges[joint]] for joint in path.edgeData]...)

function configuration_derivative!{X}(out::AbstractVector{X}, state::MechanismState{X})
    mechanism = state.mechanism
    for vertex in non_root_vertices(mechanism)
        joint = vertex.edgeToParentData
        qRange = mechanism.qRanges[joint]
        vRange = state.mechanism.vRanges[joint]
        @inbounds qjoint = view(state.q, qRange)
        @inbounds vjoint = view(state.v, vRange)
        @inbounds q̇joint = view(out, qRange)
        velocity_to_configuration_derivative!(joint, q̇joint, qjoint, vjoint)
    end
end

function configuration_derivative{X}(state::MechanismState{X})
    ret = Vector{X}(num_positions(state))
    configuration_derivative!(ret, state)
    ret
end

function setdirty!(state::MechanismState)
    for element in values(state.jointTransforms) setdirty!(element) end
    for element in values(state.transformsToRoot) setdirty!(element) end
    for element in values(state.twists) setdirty!(element) end
    for element in values(state.biasAccelerations) setdirty!(element) end
    for element in values(state.motionSubspaces) setdirty!(element) end
    for element in values(state.spatialInertias) setdirty!(element) end
    for element in values(state.crbInertias) setdirty!(element) end
end

function zero_configuration!(state::MechanismState)
    X = eltype(state.q)
    for joint in joints(state.mechanism)
        zero_configuration!(joint, configuration(state, joint))
    end
    setdirty!(state)
end

function zero_velocity!(state::MechanismState)
    X = eltype(state.v)
    fill!(state.v,  zero(X))
    setdirty!(state)
end

zero!(state::MechanismState) = begin zero_configuration!(state); zero_velocity!(state) end

function rand_configuration!(state::MechanismState)
    X = eltype(state.q)
    for joint in joints(state.mechanism)
        rand_configuration!(joint, configuration(state, joint))
    end
    setdirty!(state)
end

function rand_velocity!(state::MechanismState)
    rand!(state.v)
    setdirty!(state)
end

rand!(state::MechanismState) = begin rand_configuration!(state); rand_velocity!(state) end

function set_configuration!(state::MechanismState, joint::Joint, q::AbstractVector)
    configuration(state, joint)[:] = q
    setdirty!(state)
end

function set_velocity!(state::MechanismState, joint::Joint, v::AbstractVector)
    velocity(state, joint)[:] = v
    setdirty!(state)
end

function set_configuration!(state::MechanismState, q::AbstractVector)
    copy!(state.q, q)
    setdirty!(state)
end

function set_velocity!(state::MechanismState, v::AbstractVector)
    copy!(state.v, v)
    setdirty!(state)
end

function set!(state::MechanismState, x::AbstractVector)
    nq = num_positions(state)
    nv = num_velocities(state)
    length(x) == nq + nv || error("wrong size")
    unsafe_copy!(state.q, 1, x, 1, nq)
    unsafe_copy!(state.v, 1, x, nq + 1, nv)
    setdirty!(state)
end

function add_transform_to_root_cache!(state::MechanismState, frame::CartesianFrame3D)
    C = eltype(state)
    state.transformsToRoot[frame] = CacheElement(Transform3D(C, frame, root_frame(state.mechanism)))
    setdirty!(state)
end

function add_frame!(state::MechanismState, transform::Transform3D)
    C = eltype(state)
    state.fixedTransformsToParent[transform.from] = convert(Transform3D{C}, transform)
    add_transform_to_root_cache!(state, transform.from)
end

function add_frame!{X, M, C}(state::MechanismState{X, M, C}, joint::Joint{M}, q::VectorSegment{X}, v::VectorSegment{X})
    state.jointTransforms[joint] = CacheElement{Transform3D{C}}()
    add_transform_to_root_cache!(state, joint.frameAfter)
end

function MechanismState{X, M}(::Type{X}, m::Mechanism{M})
    C = promote_type(M, X)
    state = MechanismState{X, M, C}(m)

    state.transformsToRoot[root_frame(m)] = CacheElement{Transform3D{C}}()

    for vertex in m.toposortedTree
        body = vertex.vertexData
        if !isroot(vertex)
            joint = vertex.edgeToParentData

            qJoint = configuration(state, joint)
            vJoint = velocity(state, joint)
            jointAndState = JointAndState(joint, qJoint, vJoint)
            state.frameAfterJointToJointAndState[joint.frameAfter] = jointAndState
            state.successorBodyToJointAndState[body] = jointAndState
            add_frame!(state, m.jointToJointTransforms[joint])
            add_frame!(state, joint, qJoint, vJoint)
            state.motionSubspaces[joint] = CacheElement{GeometricJacobian}()
        end

        random_frame = body.frame # just used to initialize
        state.twists[body] = CacheElement{Twist{C}}()
        state.biasAccelerations[body] = CacheElement{SpatialAcceleration{C}}()
        state.spatialInertias[body] = CacheElement{SpatialInertia{C}}()
        state.crbInertias[body] = CacheElement{SpatialInertia{C}}()

        # additional body fixed frames
        for transform in m.bodyFixedFrameDefinitions[body]
            if transform.from != transform.to
                add_frame!(state, transform)
            end
        end
    end
    state
end

function transform_to_parent(state::MechanismState, frame::CartesianFrame3D)
    if haskey(state.frameAfterJointToJointAndState, frame)
        # joint transform
        info = state.frameAfterJointToJointAndState[frame]
        element = state.jointTransforms[info.joint]
        if element.dirty
            update!(element, joint_transform(info.joint, info.q))
        end
        ret = get(element)
    else
        # fixed transform
        ret = state.fixedTransformsToParent[frame]
    end
    ret
end

function transform_to_root(state::MechanismState, frame::CartesianFrame3D)
    element = state.transformsToRoot[frame]
    if element.dirty
        toParent = transform_to_parent(state, frame)
        toRoot = toParent.to == root_frame(state.mechanism) ? toParent : transform_to_root(state, toParent.to) * toParent
        update!(element, toRoot)
    end
    get(element)
end

function relative_transform(state::MechanismState, from::CartesianFrame3D, to::CartesianFrame3D)
    rootFrame = root_frame(state.mechanism)
    if to == rootFrame
        ret = transform_to_root(state, from)
    elseif from == rootFrame
        ret = inv(transform_to_root(state, to))
    else
        ret = inv(transform_to_root(state, to)) * transform_to_root(state, from)
    end
    ret
end

function transform(state::MechanismState, point::Point3D, to::CartesianFrame3D)
    point.frame == to && return point # nothing to be done
    relative_transform(state, point.frame, to) * point
end

function transform(state::MechanismState, vector::FreeVector3D, to::CartesianFrame3D)
    vector.frame == to && return vector # nothing to be done
    relative_transform(state, vector.frame, to) * vector
end

function transform(state::MechanismState, twist::Twist, to::CartesianFrame3D)
    twist.frame == to && return twist # nothing to be done
    transform(twist, relative_transform(state, twist.frame, to))
end

function transform(state::MechanismState, wrench::Wrench, to::CartesianFrame3D)
    wrench.frame == to && return wrench # nothing to be done
    transform(wrench, relative_transform(state, wrench.frame, to))
end

function transform(state::MechanismState, accel::SpatialAcceleration, to::CartesianFrame3D)
    accel.frame == to && return accel # nothing to be done
    oldToRoot = transform_to_root(state, accel.frame)
    rootToOld = inv(oldToRoot)
    twistOfBodyWrtBase = transform(relative_twist(state, accel.body, accel.base), rootToOld)
    twistOfOldWrtNew = transform(relative_twist(state, accel.frame, to), rootToOld)
    oldToNew = inv(transform_to_root(state, to)) * oldToRoot
    transform(accel, oldToNew, twistOfOldWrtNew, twistOfBodyWrtBase)
end

function update_twists_and_bias_accelerations!(state::MechanismState)
    C = eltype(state)
    for vertex in state.mechanism.toposortedTree
        body = vertex.vertexData
        if !isroot(vertex)
            joint = vertex.edgeToParentData
            parentBody = vertex.parent.vertexData
            parentTwist = twist_wrt_world(state, parentBody)
            jointTwist = joint_twist(joint, configuration(state, joint), velocity(state, joint))
            parentFrame = default_frame(state.mechanism, parentBody)
            jointTwist = Twist(joint.frameAfter, parentFrame, jointTwist.frame, jointTwist.angular, jointTwist.linear) # to make the frames line up;
            bodyToRoot = transform_to_root(state, joint.frameAfter)
            twist = parentTwist + transform(jointTwist, bodyToRoot)
            update!(state.twists[body], twist)

            parentBias = bias_acceleration(state, parentBody)
            jointBias = bias_acceleration(joint, configuration(state, joint), velocity(state, joint))
            jointBias = SpatialAcceleration(joint.frameAfter, parentFrame, jointBias.frame, jointBias.angular, jointBias.linear) # to make the frames line up
            twistOfBodyWrtRoot = transform(twist, inv(bodyToRoot))
            bias = parentBias + transform(jointBias, bodyToRoot, twistOfBodyWrtRoot, jointTwist) # TODO: jointTwist
            update!(state.biasAccelerations[body], bias)
        else
            rootFrame = root_frame(state.mechanism)
            update!(state.twists[body], zero(Twist{C}, rootFrame, rootFrame, rootFrame))
            update!(state.biasAccelerations[body], zero(SpatialAcceleration{C}, rootFrame, rootFrame, rootFrame))
        end
    end
end

function twist_wrt_world{X, M}(state::MechanismState{X, M}, body::RigidBody{M})
    element = state.twists[body]
    if element.dirty
        update_twists_and_bias_accelerations!(state)
    end
    get(element)
end

relative_twist{X, M}(state::MechanismState{X, M}, body::RigidBody{M}, base::RigidBody{M}) = -twist_wrt_world(state, base) + twist_wrt_world(state, body)

function relative_twist(state::MechanismState, bodyFrame::CartesianFrame3D, baseFrame::CartesianFrame3D)
    twist = relative_twist(state, state.mechanism.bodyFixedFrameToBody[bodyFrame],  state.mechanism.bodyFixedFrameToBody[baseFrame])
    Twist(bodyFrame, baseFrame, twist.frame, twist.angular, twist.linear)
end

function bias_acceleration{X, M}(state::MechanismState{X, M}, body::RigidBody{M})
    element = state.biasAccelerations[body]
    if element.dirty
        update_twists_and_bias_accelerations!(state)
    end
    get(element)
end

function update_motion_subspaces!(state::MechanismState)
    for vertex in non_root_vertices(state.mechanism)
        body = vertex.vertexData
        joint = vertex.edgeToParentData
        parentBody = vertex.parent.vertexData
        motionSubspace = motion_subspace(joint, configuration(state, joint))
        motionSubspace = transform(motionSubspace, transform_to_root(state, motionSubspace.frame))
        parentFrame = default_frame(state.mechanism, parentBody)
        motionSubspace = GeometricJacobian(motionSubspace.body, parentFrame, motionSubspace.frame, motionSubspace.angular, motionSubspace.linear) # to make frames line up
        update!(state.motionSubspaces[joint], motionSubspace)
    end
end

function motion_subspace(state::MechanismState, joint::Joint)
    element = state.motionSubspaces[joint]
    if element.dirty
        update_motion_subspaces!(state)
    end
    get(element)
end

function update_inertias!(state::MechanismState)
    vertices = non_root_vertices(state.mechanism)
    for i = length(vertices) : -1 : 1
        vertex = vertices[i]
        body = vertex.vertexData
        spatialInertia = transform(body.inertia, transform_to_root(state, body.inertia.frame))
        update!(state.spatialInertias[body], spatialInertia)

        crbInertia = spatialInertia
        for child in vertex.children
            crbInertia += crb_inertia(state, child.vertexData)
        end
        update!(state.crbInertias[body], crbInertia)
    end
end

function spatial_inertia{X, M}(state::MechanismState{X, M}, body::RigidBody{M})
    element = state.spatialInertias[body]
    if element.dirty
        update_inertias!(state)
    end
    get(element)
end

function crb_inertia{X, M}(state::MechanismState{X, M}, body::RigidBody{M})
    element = state.crbInertias[body]
    if element.dirty
        update_inertias!(state)
    end
    get(element)
end
