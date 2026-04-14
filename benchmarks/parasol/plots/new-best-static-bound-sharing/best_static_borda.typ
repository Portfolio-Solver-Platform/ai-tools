#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "128s", "256s", "512s")
#let scores = (477.49, 417.14, 435.96, 470.76, 444.30, 450.83, 435.21, 459.96, 455.79, 452.57)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval (best-static)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (405.1, 489.6),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
