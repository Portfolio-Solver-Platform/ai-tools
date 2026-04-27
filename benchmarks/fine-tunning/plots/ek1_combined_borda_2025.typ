#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "96s", "128s", "160s", "192s", "224s", "256s")
#let scores = (4040.94, 3903.80, 3946.33, 3930.16, 3922.50, 3953.34, 3956.69, 3884.47, 4011.20, 3800.66, 3867.93, 3935.02, 4008.55)

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Borda Score: ek1-8c-8s-v2 Bound-Sharing (vs.\ All Solvers)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (3752.6, 4089.0),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
