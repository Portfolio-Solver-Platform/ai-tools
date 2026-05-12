#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "128s", "256s", "512s")
#let scores = (477.86, 418.94, 437.49, 471.46, 437.20, 452.58, 436.12, 463.55, 449.91, 454.88)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval (best-static)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (407.2, 489.6),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
