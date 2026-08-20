# Data provenance

The empirical study uses annual ATP workbooks from
[Tennis-Data](http://www.tennis-data.co.uk/data.php). The default build reads
2010–2025 match results, court surface, ATP rank, and historical decimal odds.

`elo_online.data` downloads each workbook to `data/cache/`. This directory is
excluded from version control, and no raw or row-level Tennis-Data records are
committed. The repository contains only aggregate metrics and figures produced
by the evaluation.

The cleaning rules are fixed in code:

- keep matches marked `Completed`;
- exclude retirements and walkovers;
- orient players alphabetically, independently of the winner;
- use Pinnacle odds when present and the reported bookmaker average otherwise;
- remove the bookmaker overround before treating odds as probabilities; and
- predict all matches on a calendar date before updating any ratings from that
  date, because the workbooks do not contain match timestamps.

Run `python scripts/run_real_data.py` to reproduce the download and analysis.
Use of the downloaded workbooks remains subject to the source site's terms.
