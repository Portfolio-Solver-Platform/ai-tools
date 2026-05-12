#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "96s", "128s", "160s", "192s", "224s", "256s")
#let scores = (654.96, 586.92, 606.35, 601.42, 597.92, 611.11, 607.27, 553.45, 638.77, 504.68, 542.85, 586.46, 629.84)

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval (ek1-8c-8s-v2)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (474.6, 685.0),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
