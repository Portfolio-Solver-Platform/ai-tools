#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("2s", "4s", "8s", "16s", "32s", "64s", "96s", "128s", "160s", "192s", "224s", "256s")
#let deltas = (706.2, -356.3, 1331.7, -522.8, -370.0, -173.9, -670.1, -396.8, -24.9, -949.1, -827.2, -475.7)

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Time Saved vs.\ No Bound Sharing (k1-8c-8s-v1, baseline: 41,627s)],
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
