function subtree_mass{T}(base::Tree{RigidBody{T}, Joint{T}})
    result = isroot(base) ? zero(T) : base.vertexData.inertia.mass
    for child in base.children
        result += subtree_mass(child)
    end
    return result
end

mass(m::Mechanism) = subtree_mass(tree(m))

mass(state::MechanismState) = mass(state.mechanism)

function center_of_mass{X, M, C}(state::MechanismState{X, M, C}, itr)
    frame = root_body(state.mechanism).frame
    com = Point3D(frame, zeros(SVector{3, C}))
    mass = zero(C)
    for body in itr
        inertia = body.inertia
        bodyCom = center_of_mass(inertia)
        com += inertia.mass * transform(state, bodyCom, frame)
        mass += inertia.mass
    end
    com /= mass
    com
end

center_of_mass(state::MechanismState) = center_of_mass(state, non_root_bodies(state.mechanism))

function geometric_jacobian{X, M, C}(state::MechanismState{X, M, C}, path::Path{RigidBody{M}, Joint{M}})
    copysign = (motionSubspace::GeometricJacobian, sign::Int64) -> sign < 0 ? -motionSubspace : motionSubspace
    motionSubspaces = [copysign(motion_subspace(state, joint), sign)::GeometricJacobian for (joint, sign) in zip(path.edgeData, path.directions)]
    hcat(motionSubspaces...)
end

function relative_acceleration{X, M, V}(state::MechanismState{X, M}, body::RigidBody{M}, base::RigidBody{M}, v̇::AbstractVector{V})
    p = path(state.mechanism, base, body)
    J = geometric_jacobian(state, p)
    v̇path = vcat([v̇[state.mechanism.vRanges[joint]] for joint in p.edgeData]...)
    bias = -bias_acceleration(state, base) + bias_acceleration(state, body)
    SpatialAcceleration(J, v̇path) + bias
end

function kinetic_energy{X, M}(state::MechanismState{X, M}, body::RigidBody{M})
    kinetic_energy(spatial_inertia(state, body), twist_wrt_world(state, body))
end

kinetic_energy{X, M}(state::MechanismState{X, M}, itr) = sum(body::RigidBody -> kinetic_energy(state, body), itr)

kinetic_energy(state::MechanismState) = kinetic_energy(state, non_root_bodies(state.mechanism))

#TODO: gravitational
function potential_energy{X, M, C}(state::MechanismState{X, M, C})
    m = mass(state.mechanism)
    gravitationalForce = m * state.mechanism.gravitationalAcceleration
    centerOfMass = transform(state, center_of_mass(state), gravitationalForce.frame)
    -dot(gravitationalForce, centerOfMass)
 end

function mass_matrix!{X, M, C}(out::Symmetric{C, Matrix{C}}, state::MechanismState{X, M, C})
    @boundscheck size(out, 1) == num_velocities(state) || error("mass matrix has wrong size")
    @assert out.uplo == 'U'
    fill!(out.data, zero(C))
    mechanism = state.mechanism

    for vi in non_root_vertices(mechanism)
        # Hii
        jointi = vi.edgeToParentData
        nvi = num_velocities(jointi)
        if nvi > 0
            bodyi = vi.vertexData
            irange = mechanism.vRanges[jointi]
            Si = motion_subspace(state, jointi)
            Ii = crb_inertia(state, bodyi)
            F = Ii * Si
            @inbounds Hii = view(out.data, irange, irange)
            @inbounds Hii[:] = Si.angular' * F.angular + Si.linear' * F.linear

            # Hji, Hij
            vj = vi.parent
            while (!isroot(vj))
                jointj = vj.edgeToParentData
                nvj = num_velocities(jointj)
                if nvj > 0
                    jrange = mechanism.vRanges[jointj]
                    Sj = motion_subspace(state, jointj)
                    framecheck(F.frame, Sj.frame)
                    @inbounds Hji = view(out.data, jrange, irange)
                    @inbounds Hji[:] = Sj.angular' * F.angular + Sj.linear' * F.linear
                end
                vj = vj.parent
            end
        end
    end
end

function mass_matrix{X, M, C}(state::MechanismState{X, M, C})
    nv = num_velocities(state)
    ret = Symmetric(Matrix{C}(nv, nv))
    mass_matrix!(ret, state)
    ret
end

function momentum_matrix(state::MechanismState)
    hcat([crb_inertia(state, vertex.vertexData) * motion_subspace(state, vertex.edgeToParentData) for vertex in non_root_vertices(state.mechanism)]...)
end

function bias_accelerations!{T, X, M}(out::Associative{RigidBody{M}, SpatialAcceleration{T}}, state::MechanismState{X, M})
    mechanism = state.mechanism
    vertices = mechanism.toposortedTree
    gravityBias = convert(SpatialAcceleration{T}, -gravitational_spatial_acceleration(mechanism))
    for vertex in non_root_vertices(mechanism)
        body = vertex.vertexData
        out[body] = gravityBias + bias_acceleration(state, body)
    end
    nothing
end

function spatial_accelerations!{T, X, M}(out::Associative{RigidBody{M}, SpatialAcceleration{T}}, state::MechanismState{X, M}, v̇::AbstractVector)
    mechanism = state.mechanism
    vertices = mechanism.toposortedTree

    # unbiased joint accelerations + gravity
    rootBody = vertices[1].vertexData
    out[rootBody] = convert(SpatialAcceleration{T}, -gravitational_spatial_acceleration(mechanism))
    for vertex in non_root_vertices(mechanism)
        body = vertex.vertexData
        joint = vertex.edgeToParentData
        S = motion_subspace(state, joint)
        @inbounds v̇joint = view(v̇, mechanism.vRanges[joint])
        joint_accel = SpatialAcceleration(S, v̇joint)
        out[body] = out[vertex.parent.vertexData] + joint_accel
    end

    # add bias acceleration - gravity
    for vertex in non_root_vertices(mechanism)
        body = vertex.vertexData
        out[body] += bias_acceleration(state, body)
    end
    nothing
end

function newton_euler!{T, X, M, W}(
        out::Associative{RigidBody{M}, Wrench{T}}, state::MechanismState{X, M},
        accelerations::Associative{RigidBody{M}, SpatialAcceleration{T}},
        externalWrenches::Associative{RigidBody{M}, Wrench{W}} = NullDict{RigidBody{M}, Wrench{T}}())

    mechanism = state.mechanism
    vertices = mechanism.toposortedTree
    for vertex in non_root_vertices(mechanism)
        body = vertex.vertexData
        joint = vertex.edgeToParentData

        Ṫbody = accelerations[body]
        I = spatial_inertia(state, body)
        Tbody = twist_wrt_world(state, body)
        wrench = newton_euler(I, Ṫbody, Tbody)
        if haskey(externalWrenches, body)
            wrench -= transform(state, externalWrenches[body], wrench.frame)
        end
        out[body] = wrench
    end
end

"""
Note: pass in net wrenches as wrenches argument. wrenches argument is modified to be joint wrenches
"""
function joint_wrenches_and_torques!{T, X, M}(
        torquesOut::AbstractVector{T},
        netWrenchesInJointWrenchesOut::Associative{RigidBody{M}, Wrench{T}},
        state::MechanismState{X, M})
    @boundscheck length(torquesOut) == num_velocities(state) || error("torquesOut size is wrong")
    mechanism = state.mechanism
    vertices = mechanism.toposortedTree
    for i = length(vertices) : -1 : 2
        vertex = vertices[i]
        joint = vertex.edgeToParentData
        body = vertex.vertexData
        jointWrench = netWrenchesInJointWrenchesOut[body]
        if !isroot(vertex.parent)
            parentBody = vertex.parent.vertexData
            netWrenchesInJointWrenchesOut[parentBody] = netWrenchesInJointWrenchesOut[parentBody] + jointWrench # action = -reaction
        end
        jointWrench = transform(state, jointWrench, joint.frameAfter)
        @inbounds τjoint = view(torquesOut, mechanism.vRanges[joint])
        joint_torque!(joint, τjoint, configuration(state, joint), jointWrench)
    end
end

function dynamics_bias!{T, X, M, W}(
        torques::AbstractVector{T},
        biasAccelerations::Associative{RigidBody{M}, SpatialAcceleration{T}},
        wrenches::Associative{RigidBody{M}, Wrench{T}},
        state::MechanismState{X, M},
        externalWrenches::Associative{RigidBody{M}, Wrench{W}} = NullDict{RigidBody{M}, Wrench{T}}())

    bias_accelerations!(biasAccelerations, state)
    newton_euler!(wrenches, state, biasAccelerations, externalWrenches)
    joint_wrenches_and_torques!(torques, wrenches, state)
end

function inverse_dynamics!{T, X, M, V, W}(
        torquesOut::AbstractVector{T},
        jointWrenchesOut::Associative{RigidBody{M}, Wrench{T}},
        accelerations::Associative{RigidBody{M}, SpatialAcceleration{T}},
        state::MechanismState{X, M},
        v̇::AbstractVector{V},
        externalWrenches::Associative{RigidBody{M}, Wrench{W}} = NullDict{RigidBody{M}, Wrench{T}}())
    spatial_accelerations!(accelerations, state, v̇)
    newton_euler!(jointWrenchesOut, state, accelerations, externalWrenches)
    joint_wrenches_and_torques!(torquesOut, jointWrenchesOut, state)
end

# note: lots of allocations, preallocate stuff and use inverse_dynamics! for performance
function inverse_dynamics{X, M, V, W}(
        state::MechanismState{X, M},
        v̇::AbstractVector{V},
        externalWrenches::Associative{RigidBody{M}, Wrench{W}} = NullDict{RigidBody{M}, Wrench{X}}())

    T = promote_type(X, M, V, W)
    torques = Vector{T}(num_velocities(state))
    jointWrenches = Dict{RigidBody{M}, Wrench{T}}()
    accelerations = Dict{RigidBody{M}, SpatialAcceleration{T}}()
    inverse_dynamics!(torques, jointWrenches, accelerations, state, v̇, externalWrenches)
    torques
end

type DynamicsResult{M, T}
    massMatrix::Symmetric{T, Matrix{T}}
    massMatrixInversionCache::Symmetric{T, Matrix{T}}
    dynamicsBias::Vector{T}
    biasedTorques::Vector{T}
    ẋ::Vector{T}
    q̇::AbstractVector{T}
    v̇::AbstractVector{T}
    accelerations::Dict{RigidBody{M}, SpatialAcceleration{T}}
    jointWrenches::Dict{RigidBody{M}, Wrench{T}}

    function DynamicsResult(::Type{T}, mechanism::Mechanism{M})
        nq = num_positions(mechanism)
        nv = num_velocities(mechanism)
        massMatrix = Symmetric(zeros(T, nv, nv))
        massMatrixInversionCache = Symmetric(zeros(T, nv, nv))
        ẋ = zeros(T, nq + nv)
        q̇ = view(ẋ, 1 : nq)
        v̇ = view(ẋ, nq + 1 : nq + nv)
        dynamicsBias = zeros(T, nv)
        biasedTorques = zeros(T, nv)
        accelerations = Dict{RigidBody{M}, SpatialAcceleration{T}}()
        sizehint!(accelerations, length(bodies(mechanism)))
        jointWrenches = Dict{RigidBody{M}, Wrench{T}}()
        sizehint!(jointWrenches, length(bodies(mechanism)))
        new(massMatrix, massMatrixInversionCache, dynamicsBias, biasedTorques, ẋ, q̇, v̇, accelerations, jointWrenches)
    end
end

DynamicsResult{M, T}(t::Type{T}, mechanism::Mechanism{M}) = DynamicsResult{M, T}(t, mechanism)

function joint_accelerations!(out::AbstractVector, massMatrixInversionCache::Symmetric, massMatrix::Symmetric, biasedTorques::Vector)
    out[:] = massMatrix \ biasedTorques # TODO: make more efficient
    nothing
end

function joint_accelerations!(out::AbstractVector{Float64}, massMatrixInversionCache::Symmetric{Float64, Matrix{Float64}}, massMatrix::Symmetric{Float64, Matrix{Float64}}, biasedTorques::Vector{Float64})
    @inbounds copy!(out, biasedTorques)
    @inbounds copy!(massMatrixInversionCache.data, massMatrix.data)
    Base.LinAlg.LAPACK.posv!(massMatrixInversionCache.uplo, massMatrixInversionCache.data, out)
    nothing
end

function dynamics!{T, X, M, W}(out::DynamicsResult{T}, state::MechanismState{X, M}, externalWrenches::Associative{RigidBody{M}, Wrench{W}} = NullDict{RigidBody{M}, Wrench{T}}())
    configuration_derivative!(out.q̇, state)
    dynamics_bias!(out.dynamicsBias, out.accelerations, out.jointWrenches, state, externalWrenches)
    @inbounds copy!(out.biasedTorques, out.dynamicsBias) # TODO: handle input torques again
    scale!(out.biasedTorques, -1)
    mass_matrix!(out.massMatrix, state)
    out.massMatrixInversionCache = out.massMatrix
    joint_accelerations!(out.v̇, out.massMatrixInversionCache, out.massMatrix, out.biasedTorques)
    nothing
end

# Convenience function that takes a Vector argument for the state and returns a Vector,
# e.g. for use with standard ODE integrators
# Note that preallocatedState is required so that we don't need to allocate a new
# MechanismState object every time this function is called
function dynamics!{T, X, M, W}(result::DynamicsResult{T}, state::MechanismState{X, M}, stateVec::Vector{X}, externalWrenches::Associative{RigidBody{M}, Wrench{W}} = NullDict{RigidBody{M}, Wrench{T}}())
    set!(state, stateVec)
    dynamics!(result, state, externalWrenches)
    return copy(result.ẋ)
end
