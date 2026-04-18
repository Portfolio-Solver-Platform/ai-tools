#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "96s", "128s", "160s", "192s", "224s", "256s")
#let scores = (4153.69, 4052.84, 4096.05, 4144.41, 3998.68, 4021.72, 4113.89, 4030.38, 4026.99, 4144.28, 4049.16, 4032.54, 4091.22)

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Borda Score: k1-8c-8s-v1 Bound-Sharing (vs.\ All Solvers)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (3967.7, 4184.7),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
