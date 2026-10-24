
import numpy as np
from scipy.optimize import brentq

# Physical constants
Q = 1.602e-19
K = 1.381e-23


class VirtualPVPanel:
    def __init__(self):
        # Reference conditions
        self.irradiance_ref = 1000.0
        self.temp_ref = 25.0

        # Operating conditions
        self.irradiance = 1000.0
        self.temperature = 25.0

        # Cell parameters
        self.I_ph_ref = 20.0
        self.I_0 = 10e-10
        self.n = 1.3
        self.Rs_cell = 0.02
        self.Rsh_cell = 100.0

        # Panel configuration
        self.Ns = 60
        self.Rs = self.Rs_cell * self.Ns
        self.Rsh = self.Rsh_cell * self.Ns

        self.I_ph = self.I_ph_ref

    def change_weather(self, irradiance, temp):
        if irradiance < 0:
            raise ValueError("Irradiance cannot be negative.")
        if temp <= -273.15:
            raise ValueError("Temperature must be above absolute zero.")

        self.irradiance = irradiance
        self.temperature = temp
        self.I_ph = self.I_ph_ref * (
            irradiance / self.irradiance_ref
        )

    def _thermal_voltage(self):
        T_kelvin = self.temperature + 273.15
        return self.Ns * self.n * K * T_kelvin / Q

    def _implicit_equation(self, current, voltage):
        Vt = self._thermal_voltage()

        exponent = (voltage + current * self.Rs) / Vt

        # Prevent numerical overflow during root finding
        if exponent > 700:
            return -np.inf

        diode_current = self.I_0 * np.expm1(exponent)
        shunt_current = (
            voltage + current * self.Rs
        ) / self.Rsh

        return (
            self.I_ph
            - diode_current
            - shunt_current
            - current
        )

    def get_current(self, voltage):
        if voltage < 0:
            raise ValueError("Voltage cannot be negative.")

        if self.I_ph <= 0:
            return 0.0

        f_low = self._implicit_equation(0.0, voltage)
        f_high = self._implicit_equation(
            self.I_ph, voltage
        )

        # No positive-current root in this interval
        if f_low <= 0:
            return 0.0

        if f_high >= 0:
            return self.I_ph

        return brentq(
            self._implicit_equation,
            0.0,
            self.I_ph,
            args=(voltage,)
        )

    def get_power(self, voltage):
        return voltage * self.get_current(voltage)

    def sweep_iv_curve(self, v_max=40.0, num_points=500):
        voltages = np.linspace(0.0, v_max, num_points)

        currents = np.array([
            self.get_current(v) for v in voltages
        ])

        powers = voltages * currents

        return voltages, currents, powers

    def get_mpp(self, v_max=40.0, num_points=500):
        V, I, P = self.sweep_iv_curve(v_max, num_points)

        idx = np.argmax(P)

        return V[idx], I[idx], P[idx]


if __name__ == "__main__":
    panel = VirtualPVPanel()
    panel.change_weather(irradiance=1000, temp=30)

    V_mpp, I_mpp, P_mpp = panel.get_mpp(v_max=40.0)

    print(f"MPP: V={V_mpp:.3f} V")
    print(f"MPP: I={I_mpp:.3f} A")
    print(f"MPP: P={P_mpp:.3f} W")
