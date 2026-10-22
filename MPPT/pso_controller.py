
# pso_controller.py
import random


class Particle:
    def __init__(self, min_volt, max_volt):
        self.voltage = random.uniform(min_volt, max_volt)
        self.velocity = 0.0

        self.best_voltage = self.voltage
        self.best_power = float("-inf")

        self.max_velocity = 0.1 * (max_volt - min_volt)

    def update_velocity(self, global_best_voltage, inertia):
        cognitive = 1.5
        social = 1.5

        r1 = random.random()
        r2 = random.random()

        cognitive_pull = (
            cognitive * r1
            * (self.best_voltage - self.voltage)
        )

        social_pull = (
            social * r2
            * (global_best_voltage - self.voltage)
        )

        self.velocity = (
            inertia * self.velocity
            + cognitive_pull
            + social_pull
        )

        # Enforce the velocity limit EVERY iteration
        self.velocity = max(
            -self.max_velocity,
            min(self.velocity, self.max_velocity)
        )

    def update_position(self, min_volt, max_volt):
        proposed_voltage = self.voltage + self.velocity

        # Keep the particle within the allowed range
        self.voltage = max(
            min_volt,
            min(proposed_voltage, max_volt)
        )


class PSO_Controller:
    def __init__(
        self,
        num_particles,
        min_volt,
        max_volt,
        max_iterations=100
    ):
        if num_particles < 1:
            raise ValueError("num_particles must be at least 1")

        if max_volt <= min_volt:
            raise ValueError("max_volt must exceed min_volt")

        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        self.swarm = [
            Particle(min_volt, max_volt)
            for _ in range(num_particles)
        ]

        self.global_best_voltage = None
        self.global_best_power = float("-inf")

        self.min_volt = min_volt
        self.max_volt = max_volt

        self.max_iterations = max_iterations
        self.iteration = 0

    def evaluate_and_move(self, virtual_panel):
        # Evaluate the current particle positions
        for particle in self.swarm:
            power = virtual_panel.get_power(particle.voltage)

            if power > particle.best_power:
                particle.best_power = power
                particle.best_voltage = particle.voltage

            if power > self.global_best_power:
                self.global_best_power = power
                self.global_best_voltage = particle.voltage

        # Stop updating after the iteration budget is reached
        if self.iteration >= self.max_iterations:
            return self.global_best_voltage, self.global_best_power

        # Inertia decreases from 0.9 toward 0.4
        progress = self.iteration / max(
            1, self.max_iterations - 1
        )

        inertia = 0.9 - 0.5 * progress

        # Move particles using the current global best
        for particle in self.swarm:
            particle.update_velocity(
                self.global_best_voltage, inertia
            )
            particle.update_position(
                self.min_volt, self.max_volt
            )

        self.iteration += 1

        return self.global_best_voltage, self.global_best_power
