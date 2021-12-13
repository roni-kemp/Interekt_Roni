# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:29:12 2016

@author: hugo


Some Class to use in RootStemExtractor


LineDrag
"""

# draggable rectangle with the animation blit techniques; see
# http://www.scipy.org/Cookbook/Matplotlib/Animations
import matplotlib.pyplot as plt

class DraggableLines:
    lock = None  # only one can be animated at a time
    
    def __init__(self, lines, linked=[]):
        self.lines = lines
        self.line = lines[0] #Un objet ligne line, = plt.plot([],[])
        self.press = None
        self.background = None
        self.changed = False
        self.linked = linked
        
    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.line.figure.canvas.mpl_connect(
            'pick_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        #print('cool')
        'on button press we will see if the mouse is over us and store some data'
        if DraggableLines.lock is not None: return
        if event.artist not in self.lines: return
            
        self.line = event.artist
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
        'on motion we will move the rect if the mouse is over us'
        if DraggableLines.lock is not self: return
        if event.inaxes != self.line.axes: return

        self.changed = True
        data = self.line.get_data()
        if data[1][0] == data[1][1]:
            self.line.set_ydata([event.ydata]*2)
        else:
            self.line.set_xdata([event.xdata]*2)
                        
        canvas = self.line.figure.canvas
        axes = self.line.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.line)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggableLines.lock is not self:
            return

        self.press = None
        DraggableLines.lock = None

        # turn off the rect animation property and reset the background
        self.line.set_animated(False)
        
        data = self.line.get_data()
        if self.linked != [] and self.line in self.linked:
            print('linked')
            for l in self.linked:
                if data[1][0] == data[1][1]:
                    l.set_ydata([data[1][1]]*2)
                else:
                    l.set_xdata([data[0][0]]*2)
                    
                    
        self.background = None

        # redraw the full figure
        self.line.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(211)
    l, = ax.plot([1,1],[0,1], picker=5)
    ax2 = fig.add_subplot(212)
    l2, = ax2.plot([0,1],[1,1], picker=5)
    l3, = ax2.plot([0,1],[0,1])
    plt.xlim(0,2)
    plt.ylim(0,2)
    test = DraggableLines([l,l2])
    test.connect()
    
    
    plt.show()
