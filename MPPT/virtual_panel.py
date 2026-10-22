import numpy as np
from scipy.optimize import brentq

# Physical constants
Q = 1.602e-19   # Elementary charge (C)
K = 1.381e-23   # Boltzmann's constant (J/K)


class VirtualPVPanel:
    def __init__(self):
        # Reference (rating) conditions the cell parameters were given at
        self.irradiance_ref = 1000   # W/m^2
        self.temp_ref = 25           # deg C

        # Current operating conditions
        self.irradiance = 1000
        self.temperature = 25

        # Solar cell properties at reference conditions
        self.I_ph_ref = 20           # Photocurrent (A) at reference irradiance
        self.I_0 = 10e-10            # Diode reverse saturation current (A)
        self.n = 1.3                 # Diode ideality factor
        self.Rs_cell = 0.02          # Series resistance per cell (Ohms)
        self.Rsh_cell = 100          # Shunt resistance per cell (Ohms)

        # Panel configuration
        self.Ns = 60.0                # Number of cells in series
        self.Rs = self.Rs_cell * self.Ns
        self.Rsh = self.Rsh_cell * self.Ns

        # Working photocurrent, kept in sync by change_weather()
        self.I_ph = self.I_ph_ref

    def change_weather(self, irradiance, temp):
        """Update operating conditions and the parameters that depend on them."""
        self.irradiance = irradiance
        self.temperature = temp

        # Simplified model: I_ph scales ~linearly with irradiance.
        # (Ignores the smaller temperature coefficient of Isc - add it later
        # if you have alpha_isc for your cell.)
        self.I_ph = self.I_ph_ref * (irradiance / self.irradiance_ref)

    def _thermal_voltage(self):
        T_kelvin = self.temperature + 273.15
        return self.n * K * T_kelvin / Q

    def _implicit_equation(self, I, V):
        """f(I, V) = 0 defines the single-diode I-V relationship."""
        Vt = self._thermal_voltage()
        diode_current = self.I_0 * np.expm1((V + I * self.Rs) / Vt)
        shunt_current = (V + I * self.Rs) / self.Rsh
        return self.I_ph - diode_current - shunt_current - I

    def get_current(self, voltage):
        """Solve the implicit single-diode equation for current at a given voltage."""
        try:
            # Root must lie between 0 A and I_ph (short-circuit current)
            current = brentq(self._implicit_equation, 0, self.I_ph, args=(voltage,))
        except ValueError:
            # No sign change in the bracket -> voltage is beyond Voc
            current = 0.0
        return max(0.0, current)

    def get_power(self, voltage):
        return voltage * self.get_current(voltage)

    def sweep_iv_curve(self, v_max=1.0, num_points=500):
        """Return arrays of voltage, current, power over the valid range."""
        voltages = np.linspace(0, v_max, num_points)
        currents = np.array([self.get_current(v) for v in voltages])
        powers = voltages * currents

        valid = currents >= 0
        return voltages[valid], currents[valid], powers[valid]

    def get_mpp(self, v_max=1.0, num_points=500):
        """Return (V_mpp, I_mpp, P_mpp), the maximum power point."""
        V, I, P = self.sweep_iv_curve(v_max, num_points)
        idx = np.argmax(P)
        return V[idx], I[idx], P[idx]


if __name__ == "__main__":
    panel = VirtualPVPanel()
    panel.change_weather(irradiance=1000, temp=30)

    V_mpp, I_mpp, P_mpp = panel.get_mpp(v_max=1.0)
    print(f"MPP: V={V_mpp:.3f} V, I={I_mpp:.3f} A, P={P_mpp:.3f} W")