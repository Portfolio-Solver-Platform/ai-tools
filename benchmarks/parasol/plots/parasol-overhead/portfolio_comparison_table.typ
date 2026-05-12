#set page(width: auto, height: auto, margin: 1em)

#table(
  columns: 4,
  align: (left, right, right, right),
  stroke: 0.5pt,

  [*Portfolio*],
  [*n*],
  [*Median ratio*],
  [*Mean ratio*],
  [cpsat8],
  [210],
  [1.80×],
  [1.99×],
  [k1],
  [194],
  [1.26×],
  [1.40×],
  [ek1],
  [194],
  [0.97×],
  [0.97×]
)

#set par(justify: true)
*Old vs new run times for the same portfolio.* For each portfolio and year,
restrict to challenge instances that finished within the 1200 s timeout in
both runs. Old data sources:

- *cpsat8*: open-category benchmarks (`cp-sat`, 8 cores) — note this is
  cp-sat run *standalone*, not via parasol, so this row mixes harness and
  machine differences.
- *k1*: `benchmarks/portfolios/all/portfolios/k1-8c-8s-v1-YEAR/`.
- *ek1*: `benchmarks/portfolios/eligible/portfolios/ek1-8c-8s-v2/ek1-8c-8s-v2-YEAR/`.

The new data is `benchmarks/portfolios/final-portfolios/portfolios-final/`
(used to build the AI training datasets). The median ratio is new / old per
instance — values >1 indicate the final-portfolios run was slower on that
shard. cpsat8 (vs standalone cp-sat) and k1 are systematically slower in the
final-portfolios run; ek1 is essentially unchanged. Combined with the
parasol-overhead replication results, the most likely explanation is uCloud
session-to-session machine variance: each (portfolio, shard) ran on a
different uCloud allocation, and some allocations were noticeably slower
than others.
