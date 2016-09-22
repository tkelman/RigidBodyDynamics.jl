function simulate(state0::MechanismState, tspan; integrator = ode45, kwargs...)
    q0 = configuration_vector(state0)
    v0 = velocity_vector(state0)
    x0 = [q0; v0]
    T = eltype(state0)
    state = state0
    result = DynamicsResult(T, state.mechanism)
    odefun(t, x) = dynamics!(result, state, x)
    times, states = integrator(odefun, x0, tspan; kwargs...)
end
