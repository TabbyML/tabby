ALTER TABLE job_runs ADD COLUMN params TEXT;
CREATE INDEX `idx_job_runs_params` ON job_runs(params);
