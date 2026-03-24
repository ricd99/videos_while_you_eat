import great_expectations as gx
import great_expectations.expectations as gxe
from typing import Tuple, List

def validate_youtube_data(df) -> Tuple(bool, List[str]):
    """
    Comprehensive data validation for youtube dataset using Great Expectations

    This function implements data quality checls that must pass before model training. 
    """

    print("Starting data validation with Great Expectations...")

     # ── STEP 1: context ───────────────────────────────────────────────────────
    # Entry point for everything in GX. get_context() with no args creates a
    # lightweight in-memory context — nothing is written to disk.
    context = gx.get_context()
 
 
    # ── STEP 2: data source + asset + batch ───────────────────────────────────
    # GX no longer wraps DataFrames directly. You register a "data source"
    # (pandas), attach a named "asset" to it, then define a "batch" which
    # is the actual slice of data to validate.
    data_source    = context.data_sources.add_pandas("yt_source")
    data_asset     = data_source.add_dataframe_asset("yt_asset")
    batch_def      = data_asset.add_batch_definition_whole_dataframe("yt_batch")
    batch          = batch_def.get_batch(batch_parameters={"dataframe": df})


    # ── STEP 3: expectation suite ─────────────────────────────────────────────
    # A suite is a named collection of expectations — equivalent to the old
    # list of ge_df.expect_*() calls, but now explicitly grouped and named.
    suite = context.suites.add(gx.ExpectationSuite(name="yt_suite"))
 
 
    # schema — required columns
    print("   Validating schema and required columns...")
    for col in ["channel_id", "channel_name", "text"]:

        suite.add_expectation(
            gxe.ExpectColumnToExist(column=col)
        )

        suite.add_expectation(
            gxe.ExpectColumnValuesToNotBeNull(column=col)
        )


    # step 4: validation definition + run
    validation_def = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name="yt_validation",
            data=batch_def,
            suite=suite,
        )
    )
    results = validation_def.run(batch_parameters={"dataframe": df})

    # step 5: process results
    failed_expectations = []
    for r in results.results:
        if not r.success:
            failed_expectations.append(r.expectation_config.type)
 
    total_checks  = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks
 
    if results.success:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"Failed expectations: {failed_expectations}")
 
    return results.success, failed_expectations
 
    