#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let xs = (2, 4, 8, 16, 32, 64, 300)
#let deltas = (469.1, 339.9, 1927.1, 2139.7, 1574.9, 2176.6, 1216.7)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Time Saved vs.\ No Bound Sharing (cp-sat + chuffed, baseline: 53,733s)],
  ylabel: [Time saved (s)],
  xlabel: [Bound-sharing interval],
  xscale: "log",
  xaxis: (
    ticks: ((2, [2s]), (4, [4s]), (8, [8s]), (16, [16s]), (32, [32s]), (64, [64s]), (300, [300s])),
    subticks: none,
  ),

  lq.plot((2, 300), (0, 0), stroke: (dash: "dashed", paint: black)),

  lq.plot(
    xs, deltas,
    mark: "o",
    mark-size: 8pt,
    stroke: 1.5pt,
  ),
)
