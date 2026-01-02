import numpy as np
import CoolProp.CoolProp as CP

class RealGas:
    def __init__(self, fluid='Hydrogen'):
        self.fluid = fluid

    def density(self, T, P):
        """
        Calculate density in kg/m^3.
        T: Temperature in K
        P: Pressure in Pa
        """
        # CoolProp expects Pressure in Pa, Temperature in K
        # Returns density in kg/m^3
        return CP.PropsSI('D', 'T', T, 'P', P, self.fluid)

    def density_molar(self, T, P):
        """
        Calculate molar density in mol/m^3.
        T: Temperature in K
        P: Pressure in Pa
        """
        return CP.PropsSI('Dmolar', 'T', T, 'P', P, self.fluid)

class DSL_Isotherm:
    def __init__(self, n1_sat, b1_0, h1, n2_sat, b2_0, h2):
        """
        Dual-Site Langmuir Isotherm.

        n_sat: Saturation loading (mol/kg)
        b_0: Pre-exponential factor (1/Pa)
        h: Heat of adsorption (J/mol) (Positive value for exothermic adsorption in this context usually,
           but Arrhenius form is b = b0 * exp(Q/RT). If Q is positive heat of adsorption.)

        Equation: n = n1_sat * b1 * P / (1 + b1 * P) + n2_sat * b2 * P / (1 + b2 * P)
        where b = b0 * exp(h / (R * T))
        """
        self.n1_sat = n1_sat
        self.b1_0 = b1_0
        self.h1 = h1
        self.n2_sat = n2_sat
        self.b2_0 = b2_0
        self.h2 = h2
        self.R = 8.314 # J/(mol K)

    def loading(self, P, T):
        """
        Calculate equilibrium loading in mol/kg.
        P: Pressure in Pa
        T: Temperature in K
        """
        b1 = self.b1_0 * np.exp(self.h1 / (self.R * T))
        b2 = self.b2_0 * np.exp(self.h2 / (self.R * T))

        n = (self.n1_sat * b1 * P) / (1 + b1 * P) + \
            (self.n2_sat * b2 * P) / (1 + b2 * P)

        return n

    @staticmethod
    def rosi_mof5_params():
        """
        Returns a DSL_Isotherm instance with parameters estimated to match
        Rosi et al. (2003) MOF-5 data.

        Rosi et al. (2003) reported:
        - 4.5 wt% at 78 K (approx saturation or low pressure? Text says "up to 4.5... at 78 K")
        - 1.0 wt% at 298 K and 20 bar.

        Molar mass MOF-5 approx 769.9 g/mol.
        4.5 wt% -> ~22.3 mol/kg.
        1.0 wt% -> ~5.0 mol/kg.

        We assume two sites.
        """
        # Parameters fitted/estimated to match Rosi et al. points roughly.
        # These are not exact from the paper (as they weren't explicitly tabulated in snippets),
        # but derived to reproduce the key figures mentioned.

        # Saturation: 17.2 H2 per formula unit (769.9 g/mol) -> 22.34 mol/kg
        # Let's split 50/50 for two sites? Or based on 4 Zn vs 3 Linkers?
        # 4 Zn + 3 Linkers? Site 1 (Zn), Site 2 (Linker).
        # Let's assume n1_sat corresponds to Zn sites (4 per FU) and n2_sat to Linker (maybe 12? or 3?)
        # Let's try equal contribution for simplicity if not specified,
        # or proportional to 4:3?
        # Actually, "up to 4.5 wt% ... at 78 K" usually implies saturation.

        n_total = 22.34 # mol/kg
        n1_sat = n_total * 0.5
        n2_sat = n_total * 0.5

        # Heats of adsorption (Qst).
        # MOF-5 is typically around 4-6 kJ/mol.
        h1 = 6000.0 # J/mol
        h2 = 4000.0 # J/mol

        # Fit b0 to match 1.0 wt% (4.96 mol/kg) at 298 K, 20 bar (2e6 Pa).
        # And ensure high uptake at 78 K.
        # At 78 K, b is large, so loading -> n_total.
        # At 298 K, b is small.
        # n ~ (n1 * b1 + n2 * b2) * P (Henry regime approx if P low, but 20 bar is not low)

        # Trial and error or optimization could be done here.
        # Let's pick reasonable b0 values for H2 in MOFs (order of 1e-9 or 1e-10 Pa^-1).

        b1_0 = 1.0e-10
        b2_0 = 1.0e-10

        # Let's try to adjust to match 1 wt% at 298K/20bar.
        # 1 wt% = 5 mol/kg.
        # P = 20e5 Pa.
        # T = 298 K.
        # b1 = b1_0 * exp(6000/(8.314*298)) = b1_0 * 11.26
        # b2 = b2_0 * exp(4000/(8.314*298)) = b2_0 * 5.02
        # n = 11.17 * (b1*P)/(1+b1*P) + 11.17 * (b2*P)/(1+b2*P) = 5
        # If we assume b1_0 = b2_0 = b0
        # b1*P = b0 * 11.26 * 2e6 = b0 * 2.25e7
        # b2*P = b0 * 5.02 * 2e6 = b0 * 1.00e7
        # 5 = 11.17 * (x/(1+x) + y/(1+y))
        # This is soluble.

        # Using a pre-calculated estimate:
        # Adjusted to better match 1 wt% at 298K / 20 bar.
        b1_0 = 1.5e-8
        b2_0 = 1.5e-8

        return DSL_Isotherm(n1_sat, b1_0, h1, n2_sat, b2_0, h2)

def isosteric_heat(dsl_model, loading, T_list):
    """
    Calculate Isosteric Heat of Adsorption using Clausius-Clapeyron relation.
    ln(P) vs 1/T at constant n.
    Slope = -dH_ads / R
    dH_ads = - Slope * R

    This function numerically calculates P for a given n at multiple T,
    then fits ln(P) vs 1/T.
    """
    import scipy.optimize

    P_values = []

    for T in T_list:
        # Inverse Langmuir: Find P such that model.loading(P, T) = loading
        # Function to find root: f(P) = loading(P) - target_loading

        def func(P):
            return dsl_model.loading(P, T) - loading

        # Initial guess.
        # Low loading -> Linear. n = n_sat * b * P -> P = n / (n_sat * b)
        # High loading -> P increases fast.

        try:
            # Bounds for pressure: 1 Pa to 1000 bar
            res = scipy.optimize.root_scalar(func, bracket=[1e-5, 1e8], method='brentq')
            P_values.append(res.root)
        except ValueError:
            # If target loading is unreachable (e.g. > saturation), return NaN
            P_values.append(np.nan)

    # Check for NaNs
    if any(np.isnan(P_values)):
        return np.nan

    ln_P = np.log(P_values)
    inv_T = 1.0 / np.array(T_list)

    # Linear regression
    slope, intercept = np.polyfit(inv_T, ln_P, 1)

    # Clausius-Clapeyron: ln P = (dH_ads / R) * (1/T) + C
    # Slope = dH_ads / R
    # Qst = - dH_ads = - Slope * R  (Note: usually Qst is defined positive)
    # Actually, ln P = -Qst/R * (1/T) + dS/R.
    # So Slope = -Qst/R  => Qst = -Slope * R.

    Qst = -slope * dsl_model.R
    return Qst
