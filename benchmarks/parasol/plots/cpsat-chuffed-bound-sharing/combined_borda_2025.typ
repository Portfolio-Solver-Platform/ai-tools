#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "300s")
#let scores = (3551.90, 3522.68, 3560.88, 3617.55, 3642.44, 3579.61, 3635.06, 3561.09)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score: cp-sat + chuffed Bound-Sharing (vs.\ All Solvers)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (3498.7, 3666.4),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
