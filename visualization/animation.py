# contains tool for animations... as the name suggests :)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from plant.compass_walker import CompassWalker


def compass_walker_single_frame(i, plant: CompassWalker, leg, body, knee, foot, time_text, dt, step_pos_rec, x_rec):
    tr = step_pos_rec[:, i]
    x = x_rec
    time_template = 'time = %.1fs'

    thislegx = [tr[0], plant.cart_coord_hip(x[:, i], tr)[0], plant.cart_coord_swing_leg(x[:, i], tr)[0]]  # [0, x1[i], x2[i]]
    thislegy = [tr[1], plant.cart_coord_hip(x[:, i], tr)[1], plant.cart_coord_swing_leg(x[:, i], tr)[1]] #[0, y1[i], y2[i]]
    thisbodyx = [plant.cart_coord_hip(x[:, i], tr)[0]] #[0, x1[i], x2[i]]
    thisbodyy = [plant.cart_coord_hip(x[:, i], tr)[1]] #[0, y1[i], y2[i]]
    thiskneex = [plant.cart_coord_stance_leg(x[:, i], tr)[0], plant.cart_coord_swing_leg(x[:, i], tr)[0]] #[0, x1[i], x2[i]]
    thiskneey = [plant.cart_coord_stance_leg(x[:, i], tr)[1], plant.cart_coord_swing_leg(x[:, i], tr)[1]] #[0, y1[i], y2[i]]
    thisfootx = [plant.cart_coord_swing_leg(x[:, i], tr)[0], plant.cart_coord_foot(x[:, i], tr)[0]]
    thisfooty = [plant.cart_coord_swing_leg(x[:, i], tr)[1], plant.cart_coord_foot(x[:, i], tr)[1]]

    leg.set_data(thislegx, thislegy)
    body.set_data(thisbodyx, thisbodyy)
    knee.set_data(thiskneex, thiskneey)
    foot.set_data(thisfootx, thisfooty)
    time_text.set_text(time_template % (i * dt))

    return leg, body, knee, foot, time_text


class AnimatorCompassWalker:
    def __init__(self, plant: CompassWalker, dt, stance_foot_pos_rec, x_rec):
        self.plant = plant
        self.dt = dt
        self.tr = stance_foot_pos_rec
        self.x_rec = x_rec

        # filled later
        self.leg = None
        self.body = None
        self.knee = None
        self.foot = None
        self.time_text = None

    def animate(self):
        fig = plt.figure(figsize=(9, 4))
        lim = [[-1 * self.plant.length_legs, 10 * self.plant.length_legs],
               [-0.2 * self.plant.length_legs, 1.5 * self.plant.length_legs]]
        ax = fig.add_subplot(111, autoscale_on=False, ylim=lim[1], xlim=lim[0])
        ax.set_aspect('equal')
        ax.grid()
        ax.plot([-1.0 * self.plant.length_legs, 10 * self.plant.length_legs], [0, 0], '-', c="g", lw=1)
        self.leg, = ax.plot([], [], '-', lw=2)
        self.body, = ax.plot([], [], 'o', ms=8, c="r")
        self.knee, = ax.plot([], [], 'o', ms=4, c="r")
        self.foot, = ax.plot([], [], '-', lw=2)
        self.time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        ani = animation.FuncAnimation(fig, self.single_frame, np.arange(0, len(self.x_rec[0, :])),
                                      interval=self.dt*1000, blit=True)

        return ani

    def single_frame(self, i):
        tr = self.tr[:, i]
        time_template = 'time = %.1fs'

        thislegx = [tr[0], self.plant.cart_coord_hip(self.x_rec[:, i], tr)[0],
                    self.plant.cart_coord_swing_leg(self.x_rec[:, i], tr)[0]]  # [0, x1[i], x2[i]]
        thislegy = [tr[1], self.plant.cart_coord_hip(self.x_rec[:, i], tr)[1],
                    self.plant.cart_coord_swing_leg(self.x_rec[:, i], tr)[1]]  # [0, y1[i], y2[i]]
        thisbodyx = [self.plant.cart_coord_hip(self.x_rec[:, i], tr)[0]]  # [0, x1[i], x2[i]]
        thisbodyy = [self.plant.cart_coord_hip(self.x_rec[:, i], tr)[1]]  # [0, y1[i], y2[i]]
        thiskneex = [self.plant.cart_coord_stance_leg(self.x_rec[:, i], tr)[0],
                     self.plant.cart_coord_swing_leg(self.x_rec[:, i], tr)[0]]  # [0, x1[i], x2[i]]
        thiskneey = [self.plant.cart_coord_stance_leg(self.x_rec[:, i], tr)[1],
                     self.plant.cart_coord_swing_leg(self.x_rec[:, i], tr)[1]]  # [0, y1[i], y2[i]]
        thisfootx = [self.plant.cart_coord_swing_leg(self.x_rec[:, i], tr)[0], self.plant.cart_coord_foot(self.x_rec[:, i], tr)[0]]
        thisfooty = [self.plant.cart_coord_swing_leg(self.x_rec[:, i], tr)[1], self.plant.cart_coord_foot(self.x_rec[:, i], tr)[1]]

        self.leg.set_data(thislegx, thislegy)
        self.body.set_data(thisbodyx, thisbodyy)
        self.knee.set_data(thiskneex, thiskneey)
        self.foot.set_data(thisfootx, thisfooty)
        self.time_text.set_text(time_template % (i * self.dt))

        return self.leg, self.body, self.knee, self.foot, self.time_text

