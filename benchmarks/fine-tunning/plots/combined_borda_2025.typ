#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "96s", "128s", "160s", "192s", "224s", "256s")
#let scores = (4167.68, 4066.88, 4109.41, 4157.68, 3984.91, 4029.87, 4122.06, 4038.46, 4034.71, 4157.38, 4056.78, 4039.98, 4097.82)

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Borda Score: k1-8c-8s-v1 Bound-Sharing (vs.\ All Solvers)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (3948.4, 4204.2),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
