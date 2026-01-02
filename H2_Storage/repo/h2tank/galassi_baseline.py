import CoolProp.CoolProp as CP
import numpy as np

# A) EOS wrappers with caching
# Simple dictionary cache to avoid repeated calls for same P,T
# Note: P in bar, T in K
_rho_cache = {}
_cp_cache = {}
_z_cache = {}

def rho(P_bar, T_K):
    """
    Density of Hydrogen at P (bar) and T (K).
    Returns rho in kg/m^3.
    """
    key = (P_bar, T_K)
    if key in _rho_cache:
        return _rho_cache[key]

    # CoolProp uses Pa, K
    try:
        r = CP.PropsSI('D', 'P', P_bar * 1e5, 'T', T_K, 'Hydrogen')
    except:
        # Fallback or robust handling?
        # For validation, we expect valid states.
        r = np.nan

    _rho_cache[key] = r
    return r

def cp(P_bar, T_K):
    """
    Specific heat capacity (constant pressure) of Hydrogen at P (bar) and T (K).
    Returns cp in J/(kg K).
    """
    key = (P_bar, T_K)
    if key in _cp_cache:
        return _cp_cache[key]

    try:
        c = CP.PropsSI('C', 'P', P_bar * 1e5, 'T', T_K, 'Hydrogen')
    except:
        c = np.nan

    _cp_cache[key] = c
    return c

def Z(P_bar, T_K):
    """
    Compressibility factor of Hydrogen at P (bar) and T (K).
    Returns Z (dimensionless).
    """
    key = (P_bar, T_K)
    if key in _z_cache:
        return _z_cache[key]

    try:
        z_val = CP.PropsSI('Z', 'P', P_bar * 1e5, 'T', T_K, 'Hydrogen')
    except:
        z_val = np.nan

    _z_cache[key] = z_val
    return z_val

def internal_energy(P_bar, T_K):
    # U = H - PV/m = H - P/rho
    # Or just ask CoolProp for 'U'
    # PropsSI returns J/kg
    return CP.PropsSI('U', 'P', P_bar * 1e5, 'T', T_K, 'Hydrogen')

def enthalpy(P_bar, T_K):
    return CP.PropsSI('H', 'P', P_bar * 1e5, 'T', T_K, 'Hydrogen')


# B) Simulation function
def simulate_fast_fill(protocol, model_params):
    """
    Simulates fast filling of a Hydrogen tank.

    Args:
        protocol (dict): Contains 'Pin_bar', 'Pfin_bar', 't_fill_s', 'Tamb_C', 'Tini_C'.
        model_params (dict): Contains 'UA' (W/K), 'vol_m3' (m^3), 'mass_tank_kg' (optional), 'cp_tank' (optional).
                             If tank thermal mass is ignored, only gas is simulated.
                             Assuming lumped gas model as per prompt "Control-mass energy balance for the in-tank gas".

    Returns:
        results (dict): {
            'time_s': np.array,
            'T_K': np.array,
            'P_bar': np.array,
            'T_peak_K': float,
            't_peak_s': float
        }
    """

    # Unpack protocol
    P_start = protocol['Pin_bar']
    P_end = protocol['Pfin_bar']
    t_fill = protocol['t_fill_s']
    T_amb = protocol['Tamb_C'] + 273.15
    T_ini = protocol['Tini_C'] + 273.15

    # Unpack model params
    UA = model_params.get('UA', 0.0) # Heat transfer coefficient * Area
    V = model_params.get('vol_m3', 0.001) # Tank volume, default small if not provided?
                                          # Galassi 2012 likely has specific volume.
                                          # If not provided in protocol, it must be in model_params.

    # Inflow assumption: Linear Pressure Ramp
    # P(t) = P_start + (P_end - P_start) * (t / t_fill)
    # Actually, for the simulation, we need to enforce P(t) to match this ramp
    # OR we solve for Pressure based on mass inflow.
    # The prompt says: "Enforce P(t_fill)=Pfin (document the inflow schedule)"
    # A robust way is to assume a constant Mass Flow Rate that results in P_fin?
    # But we don't know the final T ahead of time.
    #
    # Easier approach for "Enforce P(t_fill)=Pfin":
    # Assume a linear pressure rise P(t).
    # Then at each step, we find the state.
    # But we need Temperature T(t) to know Density rho(t), and thus Mass m(t).
    # We have the energy equation:
    # d(mu)/dt = m_dot_in * h_in - Q_dot
    #
    # Let's use a time-stepping solver.
    # State variables: T_gas
    # Driven by: P_gas (imposed by linear ramp)
    #
    # At time t: P is known. T is state.
    # rho = rho(P, T)
    # m = rho * V
    # u = u(P, T)
    # U_total = m * u
    #
    # Balance: dU_total / dt = (dm/dt) * h_in - UA(T - T_amb)
    # where dm/dt is derived from d(rho*V)/dt = V * d(rho)/dt
    #
    # This seems consistent. We need h_in.
    # Assumption: Inflow gas is at T_inflow.
    # Often T_inflow is ambient or pre-cooled.
    # If not specified, we assume T_inflow = T_amb.
    T_inflow = model_params.get('T_inflow_K', T_amb)

    dt = 1.0 # 1 second steps? or finer?
    time_steps = np.arange(0, t_fill + dt, dt)

    T_res = []
    P_res = []
    t_res = []

    # Initial state
    P_curr = P_start
    T_curr = T_ini

    # Store initial
    T_res.append(T_curr)
    P_res.append(P_curr)
    t_res.append(0.0)

    # Ramp rate
    dP_dt = (P_end - P_start) / t_fill

    for t in time_steps[1:]:
        # Target Pressure at this step
        P_next = P_start + dP_dt * t

        # Simple Euler integration or something slightly better?
        # We need to find T_next such that energy balance is satisfied.
        # Energy balance over dt:
        # (m_next * u_next) - (m_curr * u_curr) = (m_next - m_curr) * h_in - Q_dot * dt
        #
        # m_curr = rho(P_curr, T_curr) * V
        # u_curr = u(P_curr, T_curr)
        # Q_dot approx UA * (T_curr - T_amb)
        # h_in = h(P_next??, T_inflow) -- usually h(P_inlet, T_inlet). P_inlet > P_tank.
        # But h is mostly func of T for ideal gas, for real gas P matters.
        # Usually assume h_in determined by source conditions.
        # Source pressure isn't given, assume h_in at (P_next, T_inflow) or (P_source, T_inflow).
        # Let's assume h_in evaluated at T_inflow and current tank Pressure (throttling is isenthalpic? No, filling).
        # Standard assumption: h_in = h(T_source, P_source). If P_source unknown, h(T_source, high_P) is close enough.
        # Let's use h_in = h(P_next, T_inflow) as a reasonable approx if source P not given.

        m_curr = rho(P_curr, T_curr) * V
        u_curr = internal_energy(P_curr, T_curr)

        # Predictor step
        Q_loss = UA * (T_curr - T_amb)

        h_in_val = enthalpy(P_next, T_inflow) # Approx

        # We need to solve for T_next.
        # Equation:
        # rho(P_next, T_next)*V * u(P_next, T_next) - m_curr*u_curr
        # = (rho(P_next, T_next)*V - m_curr) * h_in_val - Q_loss * dt

        # Rearrange:
        # rho(P_next, T_next)*V * (u(P_next, T_next) - h_in_val) = m_curr * (u_curr - h_in_val) - Q_loss * dt

        rhs = m_curr * (u_curr - h_in_val) - Q_loss * dt

        # Root finding for T_next
        # This is mono-variable root finding: f(T) = LHS(T) - RHS = 0

        # Range for T: likely increases. T_curr to T_curr + 50?
        # Use simple secant or bisection or just brute force with optimization if needed.
        # Since it's 1D and monotonic-ish, simple solver is fine.

        def residual(T_guess):
            r = rho(P_next, T_guess)
            u = internal_energy(P_next, T_guess)
            lhs = r * V * (u - h_in_val)
            return lhs - rhs

        # Secant method
        T0 = T_curr
        T1 = T_curr + 0.1

        # Limit iterations
        for _ in range(10):
            y0 = residual(T0)
            y1 = residual(T1)
            if abs(y1 - y0) < 1e-9:
                break
            T_new = T1 - y1 * (T1 - T0) / (y1 - y0)
            T0 = T1
            T1 = T_new
            if abs(T1 - T0) < 1e-4:
                break

        T_next = T1

        # Update
        P_curr = P_next
        T_curr = T_next

        t_res.append(t)
        P_res.append(P_curr)
        T_res.append(T_curr)

    # Convert to arrays
    time_s = np.array(t_res)
    P_bar = np.array(P_res)
    T_K = np.array(T_res)

    # Metrics
    T_peak_K = np.max(T_K)
    idx_peak = np.argmax(T_K)
    t_peak_s = time_s[idx_peak]

    return {
        'time_s': time_s,
        'T_K': T_K,
        'P_bar': P_bar,
        'T_peak_K': T_peak_K,
        't_peak_s': t_peak_s
    }

if __name__ == "__main__":
    # Simple test
    print("Testing Galassi Baseline Module...")
    try:
        r = rho(100, 300)
        print(f"Rho(100 bar, 300 K) = {r:.2f} kg/m3")

        proto = {
            "Pin_bar": 1,
            "Pfin_bar": 700,
            "t_fill_s": 180,
            "Tamb_C": 25,
            "Tini_C": 25
        }
        params = {
            "UA": 5.0,
            "vol_m3": 0.04 # Example 40L tank
        }

        res = simulate_fast_fill(proto, params)
        print(f"Simulation Complete. Peak T: {res['T_peak_K']:.2f} K at {res['t_peak_s']:.1f} s")
    except Exception as e:
        print(f"Test failed: {e}")
