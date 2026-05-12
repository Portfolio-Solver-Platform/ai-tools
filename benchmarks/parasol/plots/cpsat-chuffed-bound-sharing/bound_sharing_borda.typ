#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "300s")
#let scores = (324.82, 308.70, 331.72, 354.69, 369.26, 341.57, 364.76, 320.48)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval (cp-sat + chuffed)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (296.6, 381.4),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
