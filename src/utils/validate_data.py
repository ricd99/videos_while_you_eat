import logging

logger = logging.getLogger(__name__)


def validate_data(df):
    """
    Validate data quality using simple pandas checks.
    Checks for required columns and null values.
    """
    logger.info("Starting data validation...")
    
    required_columns = ["channel_id", "channel_name", "text"]
    failed_expectations = []
    
    # Check required columns exist
    for col in required_columns:
        if col not in df.columns:
            failed_expectations.append(f"column_missing: {col}")
    
    # Check for null values in required columns
    for col in required_columns:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                failed_expectations.append(f"null_values_in: {col}")
    
    # Determine success
    is_valid = len(failed_expectations) == 0
    total_checks = len(required_columns) * 2  # 3 cols x 2 checks (exists + nulls)
    passed_checks = total_checks - len(failed_expectations)
    failed_checks = len(failed_expectations)
    
    if is_valid:
        logger.info("Data validation PASSED: %d/%d checks successful", passed_checks, total_checks)
    else:
        logger.error("Data validation FAILED: %d/%d checks failed", failed_checks, total_checks)
        logger.error("Failed expectations: %s", failed_expectations)
    
    return is_valid, failed_expectations