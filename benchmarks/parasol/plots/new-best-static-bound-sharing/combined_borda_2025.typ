#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "128s", "256s", "512s")
#let scores = (3952.10, 3820.58, 3840.83, 3906.28, 3846.90, 3879.47, 3868.55, 3901.27, 3882.46, 3897.43)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score: best-static Bound-Sharing (vs.\ All Solvers)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (3794.3, 3978.4),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
