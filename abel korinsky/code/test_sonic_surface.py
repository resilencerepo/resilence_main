import time
import numpy as np
from SonicSurface.ControlSoftware.Python.SonicSurface import SonicSurface
from SonicSurface.ControlSoftware.Python.frametimer import FrameTimer

MOD_FREQ = 600  # Increased for better modulation
TIME_PER_POS = 0.1  # Increased to 0.1s for longer tone duration
ANGLE_MAX = 20 * (np.pi/180)

def main():
    sonic_surface = SonicSurface()
    try:
        sonic_surface.connect(-1)  # Select COM3 (e.g., enter 1)
        print("SonicSurface connected.")

        frequencies = np.array([220.00, 261.63, 329.63, 293.66, 349.23, 440.00, 329.63, 392.00, 493.88]) * 8  # Increased to *8
        angles = np.linspace(-ANGLE_MAX, ANGLE_MAX, 100)
        ticker = FrameTimer(500)
        angle_index = 0
        direction = 1
        fi = 0
        last_change = time.perf_counter()

        print("Starting sweep and melody...")
        try:
            while True:
                ticker.tick()
                current_angle = angles[angle_index]
                x = 5 * np.sin(current_angle)
                y = 5 * np.cos(current_angle)
                sonic_surface.focusAtPos(x, y, 0)
                sonic_surface.sendCommit()
                sonic_surface.switchOnOrOff(False)
                print(f"Angle: {current_angle * 180 / np.pi:.2f} degrees, Frequency: {frequencies[fi]:.2f} Hz")

                angle_index += direction
                if angle_index >= len(angles) - 1:
                    direction = -1
                elif angle_index <= 0:
                    direction = 1

                now = time.perf_counter()
                if now - last_change >= TIME_PER_POS:
                    last_change = now
                    fi = (fi + 1) % len(frequencies)
                    ticker.target_fps = frequencies[fi]
                    sonic_surface.switchOnOrOff(False)

        except KeyboardInterrupt:
            print("Stopping...")
            sonic_surface.switchOnOrOff(True)
            sonic_surface.disconnect()

    except Exception as e:
        print(f"Error: {e}")
        sonic_surface.disconnect()

if __name__ == "__main__":
    main()