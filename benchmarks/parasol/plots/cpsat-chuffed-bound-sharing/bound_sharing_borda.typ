#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "300s")
#let scores = (321.89, 317.67, 332.04, 352.61, 366.63, 339.48, 362.56, 323.13)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval (cp-sat + chuffed)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (307.9, 376.4),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
