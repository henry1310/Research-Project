# main.py
from virtual_panel import VirtualPVPanel
from pso_controller import PSO_Controller
import time

def run_simulation():
    panel = VirtualPVPanel()
    # Create 5 particles testing voltages between 0V and 24V
    pso = PSO_Controller(num_particles=5, min_volt=0, max_volt=24)

    print("Starting MPPT Simulation...")
    print("Initial Weather: 1000 W/m2, 25C (Sunny)")
    temp = 25
    # Run the simulation for 10 iterations
    for step in range(15):
        # Simulate a rapid cloud cover at step 5
        if step <5:
            print("Current temperature: "+ str(temp))
            panel.change_weather(irradiance=1000, temp=temp)
            temp = temp ++2

        elif step ==10:
            print("\n>>> CLOUD COVER DETECTED! Irradiance dropped to 200 W/m2 <<<")
            panel.change_weather(irradiance=200, temp=22)
            # Reset global best so the swarm searches for the new peak
            pso.global_best_power = 0.0 

        pso.evaluate_and_move(panel)
        print(f"Step {step+1} | Best Voltage Found: {pso.global_best_voltage:.2f}V | Max Power Output: {pso.global_best_power:.2f}W")
        time.sleep(0.5) # Pause briefly to mimic real-time sensor reading

if __name__ == "__main__":
    run_simulation()