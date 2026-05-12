#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "96s", "128s", "160s", "192s", "224s", "256s")
#let scores = (4040.73, 3904.22, 3945.75, 3928.60, 3919.89, 3952.37, 3957.57, 3884.59, 4010.43, 3797.96, 3866.04, 3931.71, 4008.21)

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Borda Score: ek1-8c-8s-v2 Bound-Sharing (vs.\ All Solvers)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (3749.4, 4089.3),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
