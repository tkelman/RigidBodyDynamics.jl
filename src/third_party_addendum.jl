function quaternion_derivative(quat, angular_velocity_in_body)
    q = quat
    ω = angular_velocity_in_body
    M = @SMatrix [
        -q.x -q.y -q.z;
         q.w -q.z  q.y;
         q.z  q.w -q.x;
        -q.y  q.x  q.w]
    M * (0.5 * ω)
end

function angular_velocity_in_body(quat, quat_derivative)
    q = quat
    MInv = @SMatrix [
     -q.x  q.w  q.z -q.y;
     -q.y -q.z  q.w  q.x;
     -q.z  q.y -q.x  q.w]
    2 * (MInv * quat_derivative)
end

# TODO: notify StaticArrays maintainer
@inline (::Type{SVector{0}}){T}(::AbstractArray{T}) = zeros(SVector{0, T})
