"""L9 feedback command: STATUS = NOT YET RUN on 235B. The diffuse closed-form o_proj edit of moe_entry/fb.py (real/shuffled/empty controls) is the intended mechanism;
its covariance dims (4096) are 30B-specific (235B o_proj input is 8192) and it needs the compressed model resident. See REPORT.md rule 5/6 rows."""
raise SystemExit("L9 feedback not implemented at 235B scale yet; see REPORT.md")
