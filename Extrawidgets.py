# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:29:12 2016

@author: Hugo Chauvet
Modified and extended by Félix Hartmann.

Some Classes implementing drawable things to use in RootStemExtractor:
 - DraggableLines
 - DraggablePoints

See the SciPy cookbook on Matplotlib animations:
# http://www.scipy.org/Cookbook/Matplotlib/Animations

"""

import matplotlib.pyplot as plt
import matplotlib.pylab as mpl

class DraggableLines:
    """Implements draggable line segments.

    Each line segment has a drag direction: either horizontal or vertical.

    A subset of the lines can be linked, i.e. they move together.

    Example
    -------
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        l1, = ax1.plot([0,0.1], [0,1], picker=5)
        l2, = ax1.plot([1,1], [0,1], picker=5)
        ax2 = fig.add_subplot(212)
        l3, = ax2.plot([0,0.1], [0,1], picker=5)
        ax1.set_xlim(-0.1, 1.1)
        ax2.set_xlim(-0.1, 1.1)
        test = DraggableLines([l1, l2, l3], directions=["h", "h", "h"], linked=[l1, l3])
        test.connect()
        plt.show()

    """
    lock = None  # only one can be animated at a time

    def __init__(self, lines, directions=None, linked=None):
        self.lines = lines  # Line2D objects: line, = plt.plot([],[])
        self.line = lines[0]
        self.press = None
        self.background = None
        self.changed = False
        # 'directions' is a list of directions associated to self.lines,
        # either 'vertical' or 'horizontal' (or simply 'v' or 'h').
        if directions is not None:
            self.directions = directions
        else:
            # Guess directions from line orientations.
            # Does not work for an oblique line.
            self.directions = []
            for line in self.lines:
                data = line.get_data()
                if data[0][0] == data[0][1]:
                    # vertical line, therefore movements in the horizontal direction
                    self.directions.append("horizontal")
                elif data[1][0] == data[1][1]:
                    # horizontal line, therefore movements in the vertical direction
                    self.directions.append("vertical")
                else:
                    # oblique line, therefore no obvious movement direction
                    raise ValueError(
                            "Oblique draggable line without assigned direction!")
        if linked is not None:
            self.linked = linked  # self.linked is a subset of self.lines
        else:
            self.linked = []

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.line.figure.canvas.mpl_connect(
            'pick_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """On button press we will see if the mouse is over us and store some data."""
        if DraggableLines.lock is not None: return
        if event.artist not in self.lines: return

        self.line = event.artist

        i = self.lines.index(self.line)
        self.direction = self.directions[i]

        data = self.line.get_data()
        # If the line is oblique, let's keep track of initial reference position.
        if data[0][0] != data[0][1] and data[1][0] != data[1][1]:
            if "h" in self.direction:   # horizontal movement
                self.ref_pos = event.mouseevent.xdata # keep track of x reference
            elif "v" in self.direction: # vertical movement
                self.ref_pos = event.mouseevent.ydata # keep track of y reference
        else:
            self.ref_pos = None

        self.changed = False
        DraggableLines.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.line.figure.canvas
        axes = self.line.axes
        self.line.set_animated(True)

        canvas.draw()
        self.background = canvas.copy_from_bbox(axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.line)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        """On motion we will move the line if the mouse is over us."""
        if DraggableLines.lock is not self:
            return
        if event.inaxes != self.line.axes:
            return

        self.changed = True

        if "h" in self.direction:   # horizontal movement
            if self.ref_pos is None: # vertical line
                self.line.set_xdata([event.xdata]*2)
            else: # oblique line
                shift = event.xdata - self.ref_pos
                data = self.line.get_data()
                self.line.set_xdata([data[0][0] + shift, data[0][1] + shift])
                self.ref_pos = event.xdata

        elif "v" in self.direction: # vertical movement
            if self.ref_pos is None: # horizontal line
                self.line.set_ydata([event.ydata]*2)
            else: # oblique line
                shift = event.ydata - self.ref_pos
                data = self.line.get_data()
                self.line.set_ydata([data[1][0] + shift, data[1][1] + shift])
                self.ref_pos = event.ydata

        canvas = self.line.figure.canvas
        axes = self.line.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.line)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        """On release we reset the press data."""
        if DraggableLines.lock is not self:
            return

        self.press = None
        DraggableLines.lock = None

        # turn off the rect animation property and reset the background
        self.line.set_animated(False)

        # synchronisation between linked lines
        data = self.line.get_data()
        if self.linked and self.line in self.linked:
            for l in self.linked:
                if "h" in self.direction:   # horizontal movement
                    l.set_xdata([data[0][0], data[0][1]])
                elif "v" in self.direction: # vertical movement
                    l.set_ydata([data[1][0], data[1][1]])

        self.background = None

        # redraw the full figure
        self.line.figure.canvas.draw()

    def disconnect(self):
        """Disconnect all the stored connection ids."""
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)


class DraggablePoints:
    """
    Implement draggable points, optionally linked by line segments.

    Example
    -------
        fig = plt.figure()
        ax = fig.add_subplot(111)
        P1, = ax.plot(0, 0, 'ro', ms=12, picker=5)
        P2, = ax.plot(1, 1, 'ro', ms=12, picker=5)
        P3, = ax.plot(0, 1, 'ro', ms=12, picker=5)
        plt.xlim(-1, 2)
        plt.ylim(-1, 2)
        test = DraggablePoints([P1, P2, P3], linked=True, closed=True)
        test.connect()
        plt.show()

    """
    lock = None  # only one can be animated at a time

    def __init__(self, points, linked=False, closed=False):
        self.points = points   # each 'point' is a 2DLine object with a single data point
        self.point = points[-1]
        self.background = None
        self.changed = False
        self.linked = linked
        self.closed = closed
        self.lines = []
        self.draw_lines()

    def draw_lines(self):
        """If the points are linked, create the lines between them."""
        # Reinitialize if needed
        while self.lines:
            self.lines.pop(0).remove()
        # Create the lines
        if self.linked and len(self.points) >= 2:
            self.couples = list(zip(self.points, self.points[1:]))
            if self.closed:
                self.couples.append([self.points[-1], self.points[0]])
            for P1, P2 in self.couples:
                line = mpl.Line2D([P1.get_data()[0], P2.get_data()[0]],
                                  [P1.get_data()[1], P2.get_data()[1]], color='k')
                self.lines.append(line)
                self.point.axes.add_line(line)

    def on_press(self, event):
        """On button press we will see if the mouse is over us and store some data."""
        if DraggablePoints.lock is not None:
            return
        for point in self.points:
            if event.artist is point:
                self.point = point
                break
        else:
            return

        self.changed = False
        DraggablePoints.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        for line in self.lines:
            line.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)
        for line in self.lines:
            axes.draw_artist(line)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        """On motion we will move the rect if the mouse is over us."""
        if DraggablePoints.lock is not self:
            return
        if event.inaxes != self.point.axes:
            return

        self.changed = True
        self.point.set_data([event.xdata, event.ydata])
        if self.lines:
            for i, (P1, P2) in enumerate(self.couples):
                self.lines[i].set_data([P1.get_data()[0], P2.get_data()[0]],
                                       [P1.get_data()[1], P2.get_data()[1]])

        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)
        for line in self.lines:
            axes.draw_artist(line)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        """On release we reset the press data."""
        if DraggablePoints.lock is not self:
            return

        DraggablePoints.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        for line in self.lines:
            line.set_animated(False)
        self.background = None

        # redraw the full figure
        self.point.figure.canvas.draw()

    def reinit(self, points):
        while self.points:
            self.points.pop(0).remove()
        while self.lines:
            self.lines.pop(0).remove()
        self.points = points
        self.point = points[-1]
        self.draw_lines()

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.point.figure.canvas.mpl_connect(
            'pick_event', self.on_press)
        self.cidmotion = self.point.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidrelease = self.point.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)

    def disconnect(self):
        """Disconnect all the stored connection ids."""
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    l1, = ax1.plot([0,0.1], [0,1], picker=5)
    l2, = ax1.plot([1,1], [0,1], picker=5)
    ax2 = fig.add_subplot(212)
    l3, = ax2.plot([0,0.1], [0,1], picker=5)
    ax1.set_xlim(-0.1, 1.1)
    ax2.set_xlim(-0.1, 1.1)
    test = DraggableLines([l1, l2, l3], directions=["h", "h", "h"], linked=[l1, l3])
    test.connect()

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     P1, = ax.plot(0, 0, 'ro', ms=12, picker=5)
#     P2, = ax.plot(1, 1, 'ro', ms=12, picker=5)
#     P3, = ax.plot(0, 1, 'ro', ms=12, picker=5)
#     plt.xlim(-1, 2)
#     plt.ylim(-1, 2)
#     test = DraggablePoints([P1, P2, P3], linked=True, closed=True)
#     test.connect()

    plt.show()
