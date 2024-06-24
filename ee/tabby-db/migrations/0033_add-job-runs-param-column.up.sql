ALTER TABLE job_runs ADD COLUMN command TEXT;
CREATE INDEX `idx_job_runs_command` ON job_runs(command);
