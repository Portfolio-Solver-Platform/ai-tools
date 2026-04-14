#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let xs = (2, 4, 8, 16, 32, 64, 128, 256, 512)
#let deltas = (1235.2, 1481.1, 2557.6, 1960.4, 1657.5, 1195.8, 903.0, 1505.3, 252.4)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Time Saved vs.\ No Bound Sharing (best-static)],
  ylabel: [Time saved (s)],
  xlabel: [Bound-sharing interval],
  xscale: "log",
  xaxis: (
    ticks: ((2, [2s]), (4, [4s]), (8, [8s]), (16, [16s]), (32, [32s]), (64, [64s]), (128, [128s]), (256, [256s]), (512, [512s])),
    subticks: none,
  ),

  lq.plot((2, 512), (0, 0), stroke: (dash: "dashed", paint: black)),

  lq.plot(
    xs, deltas,
    mark: "o",
    mark-size: 8pt,
    stroke: 1.5pt,
  ),
)
