#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "96s", "128s", "160s", "192s", "224s", "256s")
#let scores = (653.19, 603.76, 625.51, 651.95, 535.15, 569.20, 623.84, 570.73, 571.81, 645.73, 576.49, 566.24, 606.39)

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval (k1-8c-8s-v1)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (511.5, 676.8),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
