# Saved results

- `benchmark/`: original seed-42 histories, protocols, environment records,
  validation/test per-image scores, aggregate metrics and preflight outcomes.
- `transfer/`: fixed-baseline Mahimandal-to-OLI2MSI scores. No NAS results.
- `audits/mahimandal_reflectance16_review_audit/`: released-image inventory,
  source overlap, timing, cloud checks and geographic split diagnostics.
- `audits/sentinel_product_metadata_audit/`: Sentinel product coefficient checks.
- `audits/radiometry_reader_audit_v2/` and `zero_source_followup_audit_v2/`:
  original-source radiometry and stored-zero adjudication.
- `audits/sourcefill_sensitivity_v1/` and `footprint_overlap/`: post hoc cohort
  exclusions, with the original split and trained models unchanged.
- `audits/registration_holdout_audit/`: fixed sample, automatic correspondences,
  failures and blank manual-review forms; not manually confirmed ground control.
- Other audit folders record auxiliary-label provenance, OLI preprocessing and
  strict source-disjoint split feasibility.

Numerical scores are archived observations, not fresh inference. File-source
records document the fixed-baseline row filtering of mixed exports. Raw remote
service-response caches, full-resolution predictions and model states are not
included. Retain the full evidence deposit for those larger artifacts.

