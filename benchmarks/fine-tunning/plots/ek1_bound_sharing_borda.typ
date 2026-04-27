#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "96s", "128s", "160s", "192s", "224s", "256s")
#let scores = (654.39, 586.03, 606.10, 601.71, 598.95, 611.03, 606.14, 552.69, 638.54, 505.88, 543.36, 587.87, 629.31)

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval (ek1-8c-8s-v2)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (476.2, 684.1),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
