from absl import app
from absl import flags
from inputs import get_gamepad
import threading
import time

FLAGS = flags.FLAGS

# class for correcting weird gamepad numbers


class Correction:
    def __init__(self, real_max, real_min, sign):
        self.real_max = real_max
        self.real_min = real_min
        self.sign = sign


class Gamepad:
    """Interface for reading commands from Logitech F710 Gamepad.

    The control works as following:
    1) Press LB+RB at any time for emergency stop
    2) Use the left joystick for forward/backward/left/right walking.
    3) Use the right joystick for rotation around the z-axis.
    4) a correction term sometimes is needed whe the joystick behave in unexpected way
    4) correction.real_min (the real.min is the value that we read by running this executable and by reading what we get by pushing the left axis all to the left)
       correction.real_max (the real.min is the value that we read by running this executable and by reading what we get by pushing the left axis all to the right)
       correction.shift
    5) here we assume that the real range is the same for each axis and we assume that the new range will be simmetrical around zero
    """

    def __init__(self, dead_zone=0, vel_scale_x=.4, vel_scale_y=.4, vel_scale_rot=1., correction=None):
        """Initialize the gamepad controller.
        Args:
          vel_scale_x: maximum absolute x-velocity command.
          vel_scale_y: maximum absolute y-velocity command.
          vel_scale_rot: maximum absolute yaw-dot command.
        """
        # with the dead_zone we filter all the non-zero values corresponding to the central positioning of the stick
        # some gamepad display some values even when the stick is in the middle so is better to force them to zero
        self.dead_zone = dead_zone
        if correction is not None:
            self.real_max = correction.real_max
            self.real_min = correction.real_min
            self.sign = correction.sign
        else:
            self.real_max = 32768
            self.real_min = 32768
            self.sign = ['minus', 'minus', 'plus']
        self._vel_scale_x = vel_scale_x
        self._vel_scale_y = vel_scale_y
        self._vel_scale_rot = vel_scale_rot
        self._lb_pressed = False
        self._rb_pressed = False

        # Controller states
        self.vx, self.vy, self.wz = 0., 0., 0.
        self.estop_flagged = False
        self.is_running = True
        self.overwrite = False

        self.read_thread = threading.Thread(target=self.read_loop)
        self.read_thread.start()

    def read_loop(self):
        """The read loop for events.

        This funnction should be executed in a separate thread for continuous
        event recording.
        """
        while self.is_running and not self.estop_flagged:
            events = get_gamepad()
            for event in events:
                self.update_command(event)

    def update_command(self, event):
        """Update command based on event readings."""
        if event.ev_type == 'Key' and event.code == 'BTN_TL':
            self._lb_pressed = bool(event.state)
        elif event.ev_type == 'Key' and event.code == 'BTN_TR':
            self._rb_pressed = bool(event.state)
        elif event.ev_type == 'Absolute' and event.code == 'ABS_X':
            # Left Joystick L/R axis
            self.vy = self._interpolate(
                event.state, self._vel_scale_y, self.sign[0])
        elif event.ev_type == 'Absolute' and event.code == 'ABS_Y':
            # Left Joystick F/B axis; need to flip sign for consistency
            self.vx = self._interpolate(
                event.state, self._vel_scale_x, self.sign[1])
        elif event.ev_type == 'Absolute' and event.code == 'ABS_RX':
            self.wz = self._interpolate(
                event.state, self._vel_scale_rot, self.sign[2])

        if self._lb_pressed:
            self.overwrite = not self.overwrite
            self.vx, self.vy, self.wz = 0., 0., 0.

        if self._lb_pressed and self._rb_pressed:
            self.estop_flagged = True
            self.vx, self.vy, self.wz = 0., 0., 0.

    def get_command(self, time_since_reset):
        del time_since_reset  # unused
        return (self.vx, self.vy, 0), self.wz, self.estop_flagged, self.overwrite

    def stop(self):
        self.is_running = False
    # the raw_reading are taken with the abs by de facto sddumng that left is always minimum and top is always minimum

    def _interpolate(self, raw_reading, new_scale, sign):
        # from https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
        scaled_value = -new_scale + \
            ((new_scale - (-new_scale))/(self.real_max-self.real_min)) * \
            (abs(raw_reading) - self.real_min)
        # dead_zone check
        if(abs(scaled_value) < self.dead_zone):
            scaled_value = 0.0
        # sign check
        if(sign == 'minus'):
            return -scaled_value
        else:
            return scaled_value


def main(_):
    gamepad = Gamepad()
    while True:
        print("Vx: {}, Vy: {}, Wz: {}, Estop: {}".format(gamepad.vx, gamepad.vy,
                                                         gamepad.wz,
                                                         gamepad.estop_flagged))
        time.sleep(0.1)


if __name__ == "__main__":
    app.run(main)
