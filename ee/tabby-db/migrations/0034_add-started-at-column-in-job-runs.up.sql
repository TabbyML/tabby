ALTER TABLE job_runs ADD COLUMN started_at TIMESTAMP;

-- Set started_at to created_at for all job runs that have already finished.
UPDATE job_runs SET started_at = created_at WHERE exit_code IS NOT NULL;