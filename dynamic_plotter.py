import matplotlib.pyplot as plt
plt.ion()
class DynamicUpdate():
    def on_launch(self):
        # Set up plot
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        #self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [], 'o')
        # Autoscale on unknown axis and known lims on the other
        #self.ax.set_autoscaley_on(False)
        # Other stuff
        #self.ax.grid()
        #self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 20)
        self.xlabel = plt.xlabel("Frames")
        self.ylabel = plt.ylabel("Risk Factor")
        self.title = plt.title("CoVision Realtime Risk Assessment")
        self.ax.set_facecolor('xkcd:grey')
        self.figure.patch.set_facecolor("grey")
        self.lowRisk = plt.axhline(y=5, color="green")
        self.highRisk = plt.axhline(y=15, color="red")
        ...

    def on_running(self, xdata, ydata):
        # Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()