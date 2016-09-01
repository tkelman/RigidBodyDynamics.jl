immutable UpdateJointTransform{M, X}
    joint::Joint{M}
    q::VectorSegment{X}
end

function (functor::UpdateJointTransform)()
    joint_transform(functor.joint, functor.q)
end

immutable UpdateTransformToRoot{C, M, X}
    frame::CartesianFrame3D
    parentToRootCache::CacheElement{Transform3D{C}, UpdateTransformToRoot{C}}

    UpdateTransformToRoot(frame) = new(frame)
    UpdateTransformToRoot(frame, parentToRootCache) = new(frame, parentToRootCache)
end

function (functor::UpdateTransformToRoot)()
    toParent = transform_to_parent(functor.frame)
    isdefined(functor.parentToRootCache) ? get(functor.parentToRootCache) * toParent : toParent
end

type TransformCache{C<:Real, M<:Real, X<:Real}
    rootFrame::CartesianFrame3D
    jointTransformsToParent::Dict{CartesianFrame3D, CacheElement{Transform3D{C}, UpdateJointTransform{M, X}}}
    fixedTransformsToParent::Dict{CartesianFrame3D, Transform3D{C}}
    transformsToRoot::Dict{CartesianFrame3D, CacheElement{Transform3D{C}, UpdateTransformToRoot{C}}}

    function TransformCache(rootFrame)
        cache = new(rootFrame, Dict(), Dict(), Dict())
        cache.transformsToRoot[rootFrame] = CacheElement(Transform3D{C}, UpdateTransformToRoot{C}(rootFrame))
        cache
    end
end
eltype{C}(::TransformCache{C}) = C

function transform_to_parent(cache::TransformCache, frame::CartesianFrame3D) # TODO: check code_warntype
    get(cache.fixedTransformsToParent, frame, get(cache.jointTransformsToParent[frame]))
end

transform_to_root(cache::TransformCache, frame::CartesianFrame3D) = get(cache.transformsToRoot[frame])

function relative_transform(cache::TransformCache, from::CartesianFrame3D, to::CartesianFrame3D)
    fromToRoot = transform_to_root(cache, from)
    to == cache.rootFrame ? fromToRoot : inv(transform_to_root(cache, to)) * fromToRoot
end

function setdirty!(cache::TransformCache)
    for element in values(cache.jointTransformsToParent) setdirty!(element) end
    for element in values(cache.transformsToRoot) setdirty!(element) end
end

function add_transform_to_root_cache!(cache::TransformCache, frame, parentFrame)
    parentToRootCache = cache.transformsToRoot[parentFrame]
    cache.transformsToRoot[frame] = CacheElement(Transform3D{C}, UpdateTransformToRoot(frame, parentToRootCache))
    setdirty!(cache)
end

function add_frame!(cache::TransformCache, transform::Transform3D)
    C = eltype(cache)
    cache.fixedTransformsToParent[transform.from] = convert(Transform3D{C}, transform)
    add_transform_to_root_cache!(cache, transform.from, transform.to)
end

function add_frame!(cache::TransformCache, joint::Joint, q::VectorSegment)
    C = eltype(cache)
    cache.jointTransformsToParent[joint.frameAfter] = CacheElement(Transform3D{C}, UpdateJointTransform(joint, q))
    add_transform_to_root_cache!(cache, transform.from, transform.to)
end

function TransformCache{M, X}(m::Mechanism{M}, q::Vector{X})
    C = promote_type(M, X)
    cache = TransformCache{C, M, X}(root_frame(m))

    for vertex in m.toposortedTree
        body = vertex.vertexData
        if !isroot(vertex)
            joint = vertex.edgeToParentData
            add_frame!(cache, m.jointToJointTransforms[joint])
            qJoint = view(q, m.qRanges[joint])
            add_frame!(cache, joint, qJoint)
        end

        # additional body fixed frames
        for transform in m.bodyFixedFrameDefinitions[body]
            if transform.from != transform.to
                add_frame!(cache, transform)
            end
        end
    end
    cache
end
