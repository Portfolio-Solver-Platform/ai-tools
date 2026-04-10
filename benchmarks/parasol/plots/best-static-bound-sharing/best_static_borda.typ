#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s")
#let scores = (330.47, 250.64, 263.46, 316.24, 323.18, 319.06, 296.95)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval (best-static)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (234.7, 346.4),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
