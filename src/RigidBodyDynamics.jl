# __precompile__() TODO: enable once Quaternions.jl tags new version

module RigidBodyDynamics

include("tree.jl")

import Base: convert, zero, one, *, +, /, -, call, inv, get, findfirst, Random.rand, Random.rand!
import Base: hcat, show, showcompact, isapprox, dot, cross, unsafe_copy!, Array, eltype, copy
using StaticArrays
using Quaternions
using DataStructures
using LightXML
import ODE: ode45

include("util.jl")
include("third_party_addendum.jl")

include("frames.jl")
include("spatial.jl")
include("rigid_body.jl")
include("joint.jl")
include("cache_element.jl")

importall .TreeDataStructure
include("mechanism.jl")
include("transform_cache.jl")
include("mechanism_state.jl")
include("mechanism_algorithms.jl")
include("parse_urdf.jl")
include("simulate.jl")

export
    # types
    CartesianFrame3D,
    Transform3D,
    Point3D,
    FreeVector3D,
    SpatialInertia,
    RigidBody,
    Joint,
    JointType,
    QuaternionFloating,
    Revolute,
    Prismatic,
    Fixed,
    Twist,
    GeometricJacobian,
    Wrench,
    Momentum,
    MomentumMatrix,
    SpatialAcceleration,
    Mechanism,
    MechanismState,
    DynamicsResult,
    # functions
    name,
    has_defined_inertia,
    transform,
    newton_euler,
    joint_torque,
    joint_torque!,
    root_frame,
    root_vertex,
    tree,
    non_root_vertices,
    root_body,
    non_root_bodies,
    isroot,
    isleaf,
    bodies,
    toposort,
    path,
    joints,
    configuration_derivative,
    velocity_to_configuration_derivative!,
    configuration_derivative_to_velocity!,
    num_positions,
    num_velocities,
    configuration,
    velocity,
    num_cols,
    joint_transform,
    motion_subspace,
    bias_acceleration,
    spatial_inertia,
    crb_inertia,
    setdirty!,
    add_body_fixed_frame!,
    attach!,
    reattach!,
    submechanism,
    change_joint_type!,
    remove_fixed_joints!,
    rand_mechanism,
    rand_chain_mechanism,
    rand_tree_mechanism,
    rand_floating_tree_mechanism,
    configuration_vector,
    velocity_vector,
    state_vector,
    rand_configuration!,
    zero_configuration!,
    rand_velocity!,
    zero_velocity!,
    set_configuration!,
    set_velocity!,
    set!,
    zero!,
    add_frame!,
    twist_wrt_world,
    relative_twist,
    transform_to_parent,
    transform_to_root,
    relative_transform,
    mass,
    center_of_mass,
    geometric_jacobian,
    relative_acceleration,
    kinetic_energy,
    potential_energy,
    mass_matrix!,
    mass_matrix,
    momentum,
    momentum_matrix,
    momentum_rate_bias,
    inverse_dynamics!,
    inverse_dynamics,
    dynamics!,
    parse_urdf,
    simulate

end # module
