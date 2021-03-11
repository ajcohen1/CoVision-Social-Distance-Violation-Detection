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
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 20)
        self.xlabel = plt.xlabel("Frames", loc='left')
        self.ylabel = plt.ylabel("Risk Factor")
        self.title = plt.title("CoVision Realtime Risk Assessment")
        self.ax.set_facecolor('xkcd:grey')
        self.figure.patch.set_facecolor("grey")
        self.lowRisk = plt.axhline(y=5, color="green", linestyle="--", label="Low Risk Threshold")
        self.highRisk = plt.axhline(y=15, color="red", linestyle="--", label="High Risk Threshold")
        self.legend = self.ax.legend(loc='upper right', bbox_to_anchor=(1.1, -0.04), ncol=2)
        ...

    def on_running(self, xdata, ydata, minX, maxX):
        # Update data (with the new _and_ the old points)
        color = ["red", "yellow", "green"]
        selector = 0
        if (ydata[-1] >= 15):
            selector = 0
        elif (ydata[-1] <= 5):
            selector = 2
        else:
            selector = 1
        self.ax.plot(xdata[-1], ydata[-1], color=color[selector], marker='o', linestyle='none')


        # Need both of these in order to rescale
        self.ax.set_xlim(minX, maxX)
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()