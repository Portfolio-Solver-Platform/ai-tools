#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "300s")
#let scores = (3539.21, 3533.53, 3553.50, 3606.61, 3630.40, 3568.75, 3623.91, 3557.51)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score: cp-sat + chuffed Bound-Sharing (vs.\ All Solvers)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (3514.2, 3649.8),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
