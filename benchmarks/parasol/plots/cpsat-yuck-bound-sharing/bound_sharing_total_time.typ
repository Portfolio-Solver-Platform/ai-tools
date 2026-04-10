#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let xs = (2, 4, 8, 16, 32, 64)
#let deltas = (654.9, 306.6, 1217.8, 1304.8, 1172.4, -149.6)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Time Saved vs.\ No Bound Sharing],
  ylabel: [Time saved (s)],
  xlabel: [Bound-sharing interval],
  xscale: "log",
  xaxis: (
    ticks: ((2, [2s]), (4, [4s]), (8, [8s]), (16, [16s]), (32, [32s]), (64, [64s])),
    subticks: none,
  ),

  // Reference line at 0
  lq.plot((2, 32), (0, 0), stroke: (dash: "dashed", paint: black)),

  // Data
  lq.plot(
    xs, deltas,
    mark: "o",
    mark-size: 8pt,
    stroke: 1.5pt,
  ),
)
