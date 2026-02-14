"""
Batch Prediction Pipeline

Provides comprehensive batch prediction capabilities:
- Large-scale prediction processing
- Multiple input sources (database, file, API)
- Multiple output formats (database, CSV, Parquet)
- Job scheduling and monitoring
- Retry logic for failed jobs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Iterator, Tuple
from datetime import datetime, timezone
from pathlib import Path
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.utils.logging import get_logger
from src.utils.config import settings
from src.database.connection import get_db_session
from src.database.models import (
    BatchPredictionJob, BatchPredictionSchedule, BatchPredictionResult, BatchPredictionLog,
    RawTransaction, CustomerFeature
)
from src.features.store import get_feature_store
from src.features.processing import DataProcessor
from src.features.rfm import RFMCalculator

logger = get_logger(__name__)


class BatchInputReader:
    """Reads input data from various sources."""
    
    def __init__(self):
        """Initialize input reader."""
        self.logger = get_logger(f"{__name__}.BatchInputReader")
    
    def read_input(
        self,
        input_source: str,
        input_config: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """
        Read input data from specified source.
        
        Args:
            input_source: Source type ('database', 'file', 'api')
            input_config: Source-specific configuration
            
        Yields:
            Dictionary of input records
        """
        if input_source == "database":
            yield from self._read_from_database(input_config)
        elif input_source == "file":
            yield from self._read_from_file(input_config)
        elif input_source == "api":
            yield from self._read_from_api(input_config)
        else:
            raise ValueError(f"Unsupported input source: {input_source}")
    
    def _read_from_database(self, config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Read from database."""
        try:
            with get_db_session() as session:
                # Get customer IDs from config
                query = config.get("query")
                customer_ids = config.get("customer_ids")
                
                if query:
                    # Execute custom query
                    from sqlalchemy import text
                    result = session.execute(text(query))
                    for row in result:
                        yield dict(row._mapping)
                elif customer_ids:
                    # Get specific customers
                    customers = session.query(RawTransaction.customer_id).filter(
                        RawTransaction.customer_id.in_(customer_ids)
                    ).distinct().all()
                    
                    for customer in customers:
                        yield {"customer_id": customer.customer_id}
                else:
                    # Get all unique customers
                    customers = session.query(RawTransaction.customer_id).distinct().all()
                    
                    for customer in customers:
                        yield {"customer_id": customer.customer_id}
                        
        except Exception as e:
            self.logger.error(f"Error reading from database: {e}", exc_info=True)
            raise
    
    def _read_from_file(self, config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Read from file."""
        file_path = Path(config.get("file_path"))
        file_format = config.get("format", "csv")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        try:
            if file_format == "csv":
                df = pd.read_csv(file_path, chunksize=config.get("chunk_size", 1000))
                for chunk in df:
                    for _, row in chunk.iterrows():
                        yield row.to_dict()
            elif file_format == "parquet":
                df = pd.read_parquet(file_path)
                for _, row in df.iterrows():
                    yield row.to_dict()
            elif file_format == "json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for record in data:
                            yield record
                    else:
                        yield data
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
        except Exception as e:
            self.logger.error(f"Error reading from file: {e}", exc_info=True)
            raise
    
    def _read_from_api(self, config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Read from API endpoint."""
        # This would integrate with an external API
        # For now, raise not implemented
        raise NotImplementedError("API input source not yet implemented")


class BatchOutputWriter:
    """Writes batch prediction results to various destinations."""
    
    def __init__(self):
        """Initialize output writer."""
        self.logger = get_logger(f"{__name__}.BatchOutputWriter")
    
    def write_output(
        self,
        output_format: str,
        output_config: Dict[str, Any],
        results: List[Dict[str, Any]],
        job_id: int
    ) -> Dict[str, Any]:
        """
        Write results to specified output format.
        
        Args:
            output_format: Output format ('database', 'file', 'csv', 'parquet')
            output_config: Output-specific configuration
            results: List of prediction results
            job_id: Job ID for tracking
            
        Returns:
            Output metadata (path, size, etc.)
        """
        if output_format == "database":
            return self._write_to_database(results, job_id)
        elif output_format in ["file", "csv", "parquet"]:
            return self._write_to_file(output_format, output_config, results, job_id)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _write_to_database(
        self,
        results: List[Dict[str, Any]],
        job_id: int
    ) -> Dict[str, Any]:
        """Write results to database."""
        try:
            with get_db_session() as session:
                for result in results:
                    db_result = BatchPredictionResult(
                        job_id=job_id,
                        customer_id=result["customer_id"],
                        prediction=result["prediction"],
                        probability=result["probability"],
                        customer_score=result.get("customer_score"),
                        risk_level=result["risk_level"],
                        features=result.get("features"),
                        model_name=result["model_name"],
                        model_version=result["model_version"],
                        processing_time_ms=result.get("processing_time_ms"),
                        row_number=result.get("row_number")
                    )
                    session.add(db_result)
                
                session.commit()
                
                return {
                    "records_written": len(results),
                    "output_type": "database"
                }
                
        except Exception as e:
            self.logger.error(f"Error writing to database: {e}", exc_info=True)
            raise
    
    def _write_to_file(
        self,
        output_format: str,
        output_config: Dict[str, Any],
        results: List[Dict[str, Any]],
        job_id: int
    ) -> Dict[str, Any]:
        """Write results to file."""
        output_dir = Path(output_config.get("output_dir", settings.data_dir / "batch_predictions"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_format = output_format if output_format != "file" else output_config.get("format", "csv")
        
        if file_format == "csv":
            file_path = output_dir / f"batch_predictions_{job_id}_{timestamp}.csv"
            df = pd.DataFrame(results)
            df.to_csv(file_path, index=False)
            file_size = file_path.stat().st_size
            
        elif file_format == "parquet":
            file_path = output_dir / f"batch_predictions_{job_id}_{timestamp}.parquet"
            df = pd.DataFrame(results)
            df.to_parquet(file_path, index=False)
            file_size = file_path.stat().st_size
            
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        return {
            "output_path": str(file_path),
            "file_size_bytes": file_size,
            "records_written": len(results),
            "output_type": "file",
            "format": file_format
        }


class BatchPredictionProcessor:
    """Processes batch predictions."""
    
    def __init__(self):
        """Initialize batch processor."""
        self.logger = get_logger(f"{__name__}.BatchPredictionProcessor")
        self.input_reader = BatchInputReader()
        self.output_writer = BatchOutputWriter()
    
    def process_batch_job(self, job_id: int) -> Dict[str, Any]:
        """
        Process a batch prediction job.
        
        Args:
            job_id: Batch prediction job ID
            
        Returns:
            Job execution result
        """
        try:
            with get_db_session() as session:
                from src.database.repositories import BatchPredictionJobRepository
                
                job_repo = BatchPredictionJobRepository(session)
                job = job_repo.get_by_id(job_id)
                
                if not job:
                    raise ValueError(f"Batch prediction job {job_id} not found")
                
                # Update job status
                job.status = "running"
                job.started_at = datetime.now(timezone.utc)
                session.commit()
                
                self.logger.info(f"Starting batch prediction job {job_id}: {job.job_name}")
                
                # Load model
                from src.api.main import load_model_from_mlflow
                model = load_model_from_mlflow(
                    job.model_name,
                    job.model_stage
                )
                
                # Initialize feature store if enabled
                feature_store = None
                if job.use_feature_store:
                    feature_store = get_feature_store()
                
                # Read input data
                input_records = list(self.input_reader.read_input(
                    job.input_source,
                    job.input_config
                ))
                
                job.total_records = len(input_records)
                session.commit()
                
                # Process in batches
                results = []
                batch_size = job.batch_size
                total_batches = (len(input_records) + batch_size - 1) // batch_size
                
                start_time = time.time()
                
                for batch_idx in range(0, len(input_records), batch_size):
                    batch = input_records[batch_idx:batch_idx + batch_size]
                    batch_results = self._process_batch(
                        batch,
                        model,
                        feature_store,
                        job,
                        batch_idx
                    )
                    results.extend(batch_results)
                    
                    # Update progress
                    job.processed_records = len(results)
                    job.failed_records = job.total_records - job.processed_records
                    job.progress_percentage = (job.processed_records / job.total_records * 100) if job.total_records > 0 else 0.0
                    session.commit()
                    
                    self.logger.info(
                        f"Job {job_id}: Processed {job.processed_records}/{job.total_records} records "
                        f"({job.progress_percentage:.1f}%)"
                    )
                
                # Write output
                output_metadata = self.output_writer.write_output(
                    job.output_format,
                    job.output_config,
                    results,
                    job_id
                )
                
                # Calculate performance metrics
                elapsed_time = time.time() - start_time
                records_per_second = len(results) / elapsed_time if elapsed_time > 0 else 0
                
                # Update job with results
                job.status = "completed"
                job.completed_at = datetime.now(timezone.utc)
                job.output_path = output_metadata.get("output_path")
                job.output_file_size_bytes = output_metadata.get("file_size_bytes")
                job.records_per_second = records_per_second
                session.commit()
                
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "total_records": job.total_records,
                    "processed_records": job.processed_records,
                    "failed_records": job.failed_records,
                    "output_path": job.output_path,
                    "records_per_second": records_per_second
                }
                
        except Exception as e:
            self.logger.error(f"Error processing batch job {job_id}: {e}", exc_info=True)
            
            # Update job status
            try:
                with get_db_session() as session:
                    from src.database.repositories import BatchPredictionJobRepository
                    job_repo = BatchPredictionJobRepository(session)
                    job = job_repo.get_by_id(job_id)
                    if job:
                        job.status = "failed"
                        job.error_message = str(e)
                        job.error_count += 1
                        job.completed_at = datetime.now(timezone.utc)
                        session.commit()
            except:
                pass
            
            raise
    
    def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        model: Any,
        feature_store: Optional[Any],
        job: BatchPredictionJob,
        batch_start_idx: int
    ) -> List[Dict[str, Any]]:
        """Process a single batch of records."""
        results = []
        
        for idx, record in enumerate(batch):
            try:
                row_number = batch_start_idx + idx
                customer_id = record.get("customer_id")
                
                if not customer_id:
                    continue
                
                # Get features
                if feature_store:
                    features = feature_store.get_features(customer_id)
                    if not features:
                        # Compute features if not in store
                        features = self._compute_features(customer_id)
                else:
                    features = self._compute_features(customer_id)
                
                if not features:
                    self.logger.warning(f"No features found for customer {customer_id}")
                    continue
                
                # Make prediction
                feature_vector = features.get("feature_vector", [])
                if not feature_vector:
                    continue
                
                features_array = np.array([feature_vector])
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_array)[0]
                    probability = float(probabilities[1])
                    prediction = int(np.argmax(probabilities))
                else:
                    prediction = int(model.predict(features_array)[0])
                    probability = float(prediction)
                
                # Determine risk level
                if probability < settings.risk_threshold_low:
                    risk_level = "low"
                elif probability > settings.risk_threshold_high:
                    risk_level = "high"
                else:
                    risk_level = "medium"
                
                # Calculate customer score (0-1000)
                customer_score = int(probability * 1000)
                
                result = {
                    "customer_id": customer_id,
                    "prediction": prediction,
                    "probability": probability,
                    "customer_score": customer_score,
                    "risk_level": risk_level,
                    "features": features,
                    "model_name": job.model_name,
                    "model_version": job.model_version or "latest",
                    "row_number": row_number
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing record {idx}: {e}", exc_info=True)
                job.failed_records += 1
                continue
        
        return results
    
    def _compute_features(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Compute features for a customer."""
        try:
            with get_db_session() as session:
                # Get customer transactions
                transactions = session.query(RawTransaction).filter(
                    RawTransaction.customer_id == customer_id
                ).all()
                
                if not transactions:
                    return None
                
                # Convert to DataFrame
                txn_data = []
                for txn in transactions:
                    txn_data.append({
                        "customer_id": txn.customer_id,
                        "amount": float(txn.amount) if txn.amount else 0.0,
                        "transaction_start_time": txn.transaction_start_time,
                    })
                
                df = pd.DataFrame(txn_data)
                
                # Compute RFM features
                rfm_calc = RFMCalculator()
                rfm_features = rfm_calc.calculate_rfm_features(df)
                
                # Process features
                processor = DataProcessor()
                processed = processor.fit_transform(df)
                
                # Combine features
                feature_vector = processed.values[0].tolist() if len(processed) > 0 else []
                
                return {
                    "feature_vector": feature_vector,
                    "rfm_features": rfm_features
                }
                
        except Exception as e:
            self.logger.error(f"Error computing features for customer {customer_id}: {e}", exc_info=True)
            return None


def get_batch_prediction_processor() -> BatchPredictionProcessor:
    """Get a singleton BatchPredictionProcessor instance."""
    global _batch_processor
    if '_batch_processor' not in globals():
        _batch_processor = BatchPredictionProcessor()
    return _batch_processor
