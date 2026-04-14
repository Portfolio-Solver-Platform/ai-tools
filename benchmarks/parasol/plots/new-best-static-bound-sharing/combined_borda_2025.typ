#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "128s", "256s", "512s")
#let scores = (3942.65, 3808.37, 3829.19, 3896.19, 3849.05, 3867.46, 3858.04, 3885.50, 3886.16, 3884.21)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score: best-static Bound-Sharing (vs.\ All Solvers)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (3781.5, 3969.5),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
