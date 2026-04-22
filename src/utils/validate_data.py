import logging

import great_expectations as gx
import great_expectations.expectations as gxe

logger = logging.getLogger(__name__)


def validate_data(df) -> tuple[bool, list[str]]:
    logger.info("Starting data validation with Great Expectations...")

    context = gx.get_context()

    data_source = context.data_sources.add_pandas("yt_source")
    data_asset = data_source.add_dataframe_asset("yt_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("yt_batch")

    suite = context.suites.add(gx.ExpectationSuite(name="yt_suite"))

    logger.info("Validating schema and required columns...")
    for col in ["channel_id", "channel_name", "text"]:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))

    validation_def = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name="yt_validation",
            data=batch_def,
            suite=suite,
        )
    )
    results = validation_def.run(batch_parameters={"dataframe": df})

    failed_expectations = []
    for r in results.results:
        if not r.success:
            failed_expectations.append(r.expectation_config.type)

    total_checks = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks

    if results.success:
        logger.info("Data validation PASSED: %d/%d checks successful", passed_checks, total_checks)
    else:
        logger.error("Data validation FAILED: %d/%d checks failed", failed_checks, total_checks)
        logger.error("Failed expectations: %s", failed_expectations)

    return results.success, failed_expectations