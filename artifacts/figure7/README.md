After running `run_all.sh`, the `figure7` directory should contain the following files:

```bash
figure7
├── 0_figure7_lstm.tsv
├── 1_figure7_dilated.tsv
├── 2_figure7_grid.tsv
├── 3_figure7_b2b_gemm.tsv
├── 4_figure7_attention.tsv
├── 5_figure7_attention.tsv
├── 6_figure7_bigbird.tsv
├── README.md
├── post_process_log.py
└── run_all.sh
```

The naming of the processed logs follows the pattern: `N_figure7_x.tsv`, where `N` indicates the subplot number (counting from top left to bottom right) in Figure 7 of the paper.
