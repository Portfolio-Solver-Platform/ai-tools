#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let xs = (2, 4, 8, 16, 32)
#let deltas = (656.6, 308.0, 1219.0, 1306.3, 1173.3)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Time Saved vs.\ No Bound Sharing],
  ylabel: [Time saved (s)],
  xlabel: [Bound-sharing interval],
  xscale: "log",
  xaxis: (
    ticks: ((2, [2s]), (4, [4s]), (8, [8s]), (16, [16s]), (32, [32s])),
    subticks: none,
  ),

  // Reference line at 0
  lq.plot((2, 32), (0, 0), stroke: (dash: "dashed", paint: gray)),

  // Data
  lq.plot(
    xs, deltas,
    mark: "o",
    mark-size: 8pt,
    stroke: 1.5pt,
  ),
)
