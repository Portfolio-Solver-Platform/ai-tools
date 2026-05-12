#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("2s", "4s", "8s", "16s", "32s", "64s", "96s", "128s", "160s", "192s", "224s", "256s")
#let deltas = (702.8, -359.5, 1328.6, -525.9, -373.2, 366.3, -673.2, -399.7, -28.4, -951.8, -830.2, -479.2)

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Time Saved vs.\ No Bound Sharing (k1-8c-8s-v1, baseline: 41,620s)],
  ylabel: [Time saved (s)],
  xlabel: [Bound-sharing interval],
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.plot((0, 11), (0, 0), stroke: (dash: "dashed", paint: black)),

  lq.plot(
    range(labels.len()), deltas,
    mark: "o",
    mark-size: 8pt,
    stroke: 1.5pt,
  ),
)
