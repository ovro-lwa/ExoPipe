"""Common substrate (§2): config, exclusions, astrometry, products,
residuals, qa, validate. Nothing here may be duplicated or reimplemented
inside a capability. exclusions is the ONLY reader of data_decisions.csv;
every position/association flows through astrometry (two-tier contract,
§5); validate gates REFUSE, never warn (§11.1)."""
